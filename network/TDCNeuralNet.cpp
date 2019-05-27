/*
 * TDNeuralNet.cpp
 *
 *  Created on: Jan 14, 2018
 *      Author: kfedrick
 */

#include <TDCNeuralNet.h>

namespace flexnnet
{

   TDCNeuralNet::TDCNeuralNet (const char *_name) :
      BaseNeuralNet (_name)
   {
      hv_initialized_flag = false;
   }

   TDCNeuralNet::TDCNeuralNet (const string &_name = "TDCNeuralNet") :
      BaseNeuralNet (_name)
   {
      hv_initialized_flag = false;
   }

   TDCNeuralNet::~TDCNeuralNet ()
   {
   }

   void TDCNeuralNet::resize_history (unsigned int sz)
   {
      BaseNeuralNet::resize_history (sz);

      const vector<BaseLayer *> &network_layers = get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         RdEdy_map[name] = vector<double> (output_sz);
         RdEdx_map[name] = vector<double> (output_sz);

         Ry_map[name] = vector<double> (output_sz);
         Rx_map[name] = vector<double> (output_sz);

         v_map[name] = Array<double> ();
         v_map[name].resize (output_sz + 1, input_sz);

         Hv_map[name] = Array<double> ();
         Hv_map[name].resize (output_sz + 1, input_sz);
      }
   }

   void TDCNeuralNet::init_tdcnet ()
   {
      map<const BaseLayer *, ConnectionMap *> &conn_map = get_connection_map ();

      for (std::map<const BaseLayer *, ConnectionMap *>::iterator iter =
         conn_map.begin (); iter != conn_map.end (); ++iter)
      {
         const BaseLayer *k = iter->first;
         const ConnectionMap *cm = iter->second;

         hv_conn_map[k] = new HvMap (*cm, Ry_map);
      }

      hv_initialized_flag = true;

      const vector<BaseLayer *> network_layers = get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         unsigned int sz = layer.size ();
         unsigned int isz = layer.get_input_error ().size ();

         Ry_map[name] = vector<double> (sz, 0.0);
         Rx_map[name] = vector<double> (isz, 0.0);
         RdEdy_map[name] = vector<double> (sz, 0.0);
         RdEdx_map[name] = vector<double> (sz, 0.0);

         v_map.insert (pair<string, Array<double>> (name, Array<double> (sz, isz + 1)));
         Hv_map.insert (pair<string, Array<double>> (name, Array<double> (sz, isz + 1)));

         v_map[name] = 0;
         Hv_map[name] = 0;
      }
   }

   void TDCNeuralNet::clear_hv (void)
   {
      const vector<BaseLayer *> &network_layers = get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();

         Hv_map[name] = 0;

         Ry_map[name].assign (output_sz, 0.0);
         Rx_map[name].assign (output_sz, 0.0);

         RdEdy_map[name].assign (output_sz, 0.0);
         RdEdx_map[name].assign (output_sz, 0.0);
      }
   }

   void TDCNeuralNet::set_v (const string &_name, const Array<double> &_v)
   {
      v_map[_name] = _v;
   }

   const Array<double> &TDCNeuralNet::get_Hv (const string &_name)
   {
      return (Hv_map[_name]);
   }

   void TDCNeuralNet::clear_error (unsigned int timeStep)
   {
      BaseNeuralNet::clear_error (timeStep);

      for (int i = layer_activation_order.size () - 1; i >= 0; i--)
      {
         BaseLayer &layer = *layer_activation_order[i];
         const string &name = layer.name ();
         unsigned int sz = layer.size ();

         Rx_map[name].assign (sz, 0.0);
         Ry_map[name].assign (sz, 0.0);
         RdEdy_map[name].assign (sz, 0.0);
         RdEdx_map[name].assign (sz, 0.0);
      }
   }

   const Pattern &TDCNeuralNet::operator() (const Pattern &ipattern,
                                            unsigned int timeStep)
   {
      if (!topology_initialized_flag)
      {
         update_topology ();
      }

      if (!hv_initialized_flag)
         init_tdcnet ();

      return BaseNeuralNet::operator() (ipattern, timeStep);
   }

   const PatternSequence &TDCNeuralNet::operator() (const PatternSequence &ipattseq,
                                                    unsigned int startTimeStep)
   {
      if (!topology_initialized_flag)
      {
         update_topology ();
      }

      if (!hv_initialized_flag)
         init_tdcnet ();

      if (recurrent_network_flag)
         resize_history (startTimeStep + ipattseq.size ());

      network_output_patternseq.clear ();
      network_output_patternseq.resize (ipattseq.size ());

      /*
       * Iterate through all the Patterns in this sequence
       */
      unsigned int time_step = startTimeStep;
      for (unsigned int pattern_ndx = 0; pattern_ndx < ipattseq.size ();
           pattern_ndx++)
      {
         const Pattern &ipatt = ipattseq.at (pattern_ndx);

         /*
          * Activate all network layers
          */
         for (unsigned int i = 0; i < layer_activation_order.size (); i++)
         {
            // Get a network layer
            BaseLayer *layer = layer_activation_order[i];

            // Get the LayerInputFacade for this network layer from the connection map
            map<const BaseLayer *, ConnectionMap *>::iterator map_entry =
               layer_input_conn_map.find (layer);
            ConnectionMap *layer_input = map_entry->second;

            Array<double> &v = v_map[layer->name ()];
            const Array<double> &w = layer->get_weights ();
            const Array<double> &dAdN = layer->get_dAdN ();
            vector<double> &rx = Rx_map[layer->name ()];
            vector<double> &ry = Ry_map[layer->name ()];

            // Activate the network layer with the raw input from it's layer input connection map
            const vector<double> &invec = (*layer_input) (ipatt, startTimeStep, 1);
            layer->activate (invec, time_step);

            /*
             * Forward pass Ry
             */
            map<const BaseLayer *, HvMap *>::iterator hv_map_entry =
               hv_conn_map.find (layer);
            HvMap *hv_map = hv_map_entry->second;

            const vector<double> &virtual_ry_in = (*hv_map) ();

            for (unsigned int i = 0; i < layer->size (); i++)
            {
               double sumRx = v.at (i, 0); // for the bias
               for (unsigned int j = 0; j < invec.size (); j++)
               {
                  sumRx += w.at (i, j) * virtual_ry_in.at (j)
                           + v.at (i, j + 1) * invec.at (j);
               }
               rx.at (i) = sumRx;
               ry.at (i) = sumRx * dAdN.at (i, i);
            }

         }

         network_output_pattern = network_output_map (ipatt, time_step, 0);
         network_output_patternseq.at (pattern_ndx) = network_output_pattern;
      }

      return network_output_patternseq;
   }

   void TDCNeuralNet::backprop (const vector<double> &isse, unsigned int timeStep)
   {
      if (!topology_initialized_flag)
      {
         update_topology ();
      }

      if (!hv_initialized_flag)
         init_tdcnet ();

      /*
       * Backprop network error to output layers
       */
      const vector<vector<double> > &network_errorv = network_output_map.get_error (
         isse);

      const vector<ConnectionEntry> &network_output_connvec =
         network_output_map.get_input_connections ();
      for (int conn_ndx = 0; conn_ndx < network_output_connvec.size (); conn_ndx++)
      {
         const ConnectionEntry &conn = network_output_connvec.at (conn_ndx);
         BaseLayer &layer = conn.get_input_layer ();

         layer.backprop (network_errorv.at (conn_ndx), timeStep);

         /*
          * Backprop RdEdy
          */
         const vector<double> &ry = Ry_map.at (layer.name ());
         vector<double> &RdEdy = RdEdy_map.at (layer.name ());

         //RdEdy = ry;
         RdEdy.assign (RdEdy.size (), 0.0);
      }

      // cout << "closed loop steps for time " << timeStep << " = " << closed_loop_steps.at(timeStep) << endl;
      bool done = false;
      for (unsigned int closed_loop_step = closed_loop_steps.at (timeStep); !done;
           closed_loop_step--)
      {
         /*
          * Backprop error through network in reverse order of the layer activation ordering
          */
         for (int i = layer_activation_order.size () - 1; i >= 0; i--)
         {
            BaseLayer &layer = *layer_activation_order[i];

            // Backprop error through this layer to the layer inputs
            layer.backprop (timeStep);


            /*
             * Backprop RdEdx and Hv
             */

            const vector<double> &layer_errorv = layer.get_error ();
            const vector<double> &RdEdy = RdEdy_map.at (layer.name ());
            const vector<double> &rx = Rx_map.at (layer.name ());
            const Array<double> &dAdN = layer.get_dAdN ();
            const vector<double> &d2AdN = layer.get_d2AdN ();

            vector<double> &RdEdx = RdEdx_map.at (layer.name ());

            for (unsigned int i = 0; i < layer.size (); i++)
               RdEdx.at (i) = dAdN.at (i, i) * RdEdy.at (i)
                              + rx.at (i) * d2AdN.at (i) * layer_errorv.at (i);


            // backprop error at the inputs to layers providing input to this layer
            backprop_scatter (layer, timeStep, closed_loop_step);
         }

         if (closed_loop_step == 0)
            done = true;
      }
   }

   void TDCNeuralNet::backprop_scatter (BaseLayer &layer, unsigned int timeStep,
                                        unsigned int closedLoopStep)
   {
      unsigned int layer_size = layer.size ();
      unsigned int layer_input_size = layer.get_input_error ().size ();

      map<const BaseLayer *, ConnectionMap *>::iterator map_entry =
         layer_input_conn_map.find (&layer);
      ConnectionMap *layer_input = map_entry->second;

      const vector<vector<double> > &errorv = layer_input->get_error (timeStep);
      const vector<ConnectionEntry> &connvec =
         layer_input->get_input_connections ();

      for (int conn_ndx = 0; conn_ndx < connvec.size (); conn_ndx++)
      {
         const ConnectionEntry &conn = connvec.at (conn_ndx);

         const vector<double> &layer_errorv = layer.get_error ();
         const vector<double> &RdEdx = RdEdx_map.at (layer.name ());
         const Array<double> &dAdN = layer.get_dAdN ();
         const Array<double> &dAdB = layer.get_dAdB ();
         const Array<double> &w = layer.get_weights ();
         const Array<double> &v = v_map.at (layer.name ());
         const vector<double> &in_y = layer.get_input ();

         Array<double> &Hv = Hv_map.at (layer.name ());
         double dedx;

         if (conn.is_input_connection ())
         {
            unsigned int ipatt_ndx = conn.get_input_pattern_index ();

            vector<double> &errvec = network_input_error.at (ipatt_ndx);

            for (unsigned int vec_ndx = 0; vec_ndx < errvec.size (); vec_ndx++)
               errvec.at (vec_ndx) += errorv.at (conn_ndx).at (vec_ndx);

            for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
            {
               //dedx = layer_errorv.at(netin_ndx) * dAdB.at(netin_ndx, netin_ndx);
               for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
                  dedx += layer_errorv.at (out_ndx) * dAdN.at (out_ndx, netin_ndx);

               Hv.at (netin_ndx, 0) = 1 * RdEdx.at (netin_ndx) + 0 * dedx;
               for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
                  Hv.at (netin_ndx, in_ndx + 1) = in_y.at (in_ndx) * RdEdx.at (netin_ndx) + 0 * dedx;
            }

            continue;
         }

         unsigned int bpTimeStep = timeStep;
         if (conn.is_recurrent () && closedLoopStep == 0)
            bpTimeStep--;

         BaseLayer &in_layer = conn.get_input_layer ();

         in_layer.backprop (errorv.at (conn_ndx), bpTimeStep);
         unsigned int in_layer_input_size = in_layer.get_input_error ().size ();

         /*
          * Backscatter RdEdy
          */
         const vector<double> &in_Ry = Ry_map.at (in_layer.name ());
         vector<double> &in_RdEdy = RdEdy_map.at (in_layer.name ());

         for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
         {
            //dedx = layer_errorv.at(netin_ndx) * dAdB.at(netin_ndx, netin_ndx);
            for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
               dedx += layer_errorv.at (out_ndx) * dAdN.at (out_ndx, netin_ndx);

            Hv.at (netin_ndx, 0) = RdEdx.at (netin_ndx) + 0 * dedx;
            for (unsigned int in_ndx = 0; in_ndx < in_layer.size (); in_ndx++)
            {
               in_RdEdy.at (in_ndx) += w.at (netin_ndx, in_ndx) * RdEdx.at (netin_ndx)
                                       + v.at (netin_ndx, in_ndx + 1) * dedx;

               Hv.at (netin_ndx, in_ndx + 1) =
                  in_y.at (in_ndx) * RdEdx.at (netin_ndx) + in_Ry.at (in_ndx) * dedx;
            }
         }
      }
   }

} /* namespace flexnnet */
