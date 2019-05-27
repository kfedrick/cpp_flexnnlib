/*
 * TDTrainer.h
 *
 *  Created on: Feb 26, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_TEMPDIFFTRAINER3_H_
#define FLEX_NEURALNET_TEMPDIFFTRAINER3_H_

#include "BaseTrainer.h"
#include <cfloat>
#include <cmath>

namespace flexnnet
{

   template<class EFunc, class LRPolicy>
   class TemporalDiffTrainer3 : public BaseTrainer
   {
   public:
      TemporalDiffTrainer3 (BaseNeuralNet &_net);
      virtual ~TemporalDiffTrainer3 ();

      void alloc_network_learning_rates ();

      LRPolicy &get_learning_rates ();

      void set_predict_final_cost ();
      void set_predict_cumulative_cost ();

      void set_lambda (double l);
      void set_gamma (double l);

      virtual void train (const DataSet<Exemplar<Pattern, Pattern> > &trnset,
                          const DataSet<Exemplar<Pattern, Pattern> > &vldset = DataSet<
                             Exemplar<Pattern, Pattern> > (),
                          const DataSet<Exemplar<Pattern, Pattern> > &tstset = DataSet<
                             Exemplar<Pattern, Pattern> > ());

      void train (
         const DataSet<Exemplar<PatternSequence, PatternSequence> > &_trnset,
         const DataSet<Exemplar<PatternSequence, PatternSequence> > &_vldset =
         DataSet<Exemplar<PatternSequence, PatternSequence> > (),
         const DataSet<Exemplar<PatternSequence, PatternSequence> > &_tstset =
         DataSet<Exemplar<PatternSequence, PatternSequence> > ());

      double sim (
         const DataSet<Exemplar<PatternSequence, PatternSequence> > &_dataset);
      TrainingRecord::Entry &calc_performance_data (unsigned int _epoch,
                                                    const DataSet<Exemplar<PatternSequence, PatternSequence> > &_trnset,
                                                    const DataSet<Exemplar<PatternSequence,
                                                                           PatternSequence> > &_vldset =
                                                    DataSet<Exemplar<PatternSequence, PatternSequence> > (),
                                                    const DataSet<Exemplar<PatternSequence,
                                                                           PatternSequence> > &_tstset =
                                                    DataSet<Exemplar<PatternSequence, PatternSequence> > ());

   protected:

      void init_train (
         const DataSet<Exemplar<PatternSequence, PatternSequence> > &_trnset,
         const DataSet<Exemplar<PatternSequence, PatternSequence> > &_vldset =
         DataSet<Exemplar<PatternSequence, PatternSequence> > (),
         const DataSet<Exemplar<PatternSequence, PatternSequence> > &_tstset =
         DataSet<Exemplar<PatternSequence, PatternSequence> > ());

      void init_training_epoch ();

      double train_exemplar (
         const Exemplar<PatternSequence, PatternSequence> &_exemplar);

      BaseNeuralNet &get_neuralnet ();
      EFunc &get_errorfunc ();

   private:
      void td_backprop (const vector<double> &_errorv);
      void backprop_scatter (const vector<double> &ierrorv, BaseLayer &layer,
                             unsigned int timeStep);

      void save_network_gradients ();
      void accumulate_network_gradients ();
      void zero_accumulated_network_gradients ();
      void print_accumulated_network_gradients ();

      void calc_layer_bias_adj (const vector<double> &errorv,
                                const BaseLayer &layer, unsigned int timeStep,
                                vector<double> &biasDelta);
      void calc_layer_weight_adj (const vector<double> &errorv,
                                  const BaseLayer &layer, unsigned int timeStep,
                                  Array<double> &weightDelta);

   private:
      enum Prediction_Mode
      {
         FINAL_COST, CUMULATIVE_COST
      };

      /* ***********************************************
       *    Error function
       */
      EFunc error_func;

      /* ***********************************************
       *    The network
       */
      BaseNeuralNet &neural_network;

      /* ***********************************************
       *    Learning rate parameters
       */
      double lambda;
      double gamma;
      Prediction_Mode predict_mode;

      map<string, Array<double> > cumulative_dAdB_map;
      map<string, Array<double> > cumulative_dAdN_map;
      map<string, Array<double> > cumulative_dNdW_map;
      map<string, Array<double> > cumulative_dNdI_map;
      map<string, Array<double> > cumulative_dAdW_map;

      map<string, Array<double> > prev_cumulative_dAdB_map;
      map<string, Array<double> > prev_cumulative_dAdN_map;
      map<string, Array<double> > prev_cumulative_dNdW_map;
      map<string, Array<double> > prev_cumulative_dNdI_map;
      map<string, Array<double> > prev_cumulative_dAdW_map;

      unsigned int pattern_no;
   };

   template<class EFunc, class LRPolicy>
   TemporalDiffTrainer3<EFunc, LRPolicy>::TemporalDiffTrainer3 (BaseNeuralNet &_net) :
      neural_network (_net), BaseTrainer (_net)
   {
      set_batch_mode ();
      set_predict_final_cost ();
      set_random_order_mode (true);
      set_verbose (false);
      set_global_learning_rate (0.05);
      set_lambda (0.25);
      set_gamma (0.95);

   }

   template<class EFunc, class LRPolicy>
   TemporalDiffTrainer3<EFunc, LRPolicy>::~TemporalDiffTrainer3 ()
   {
      // TODO Auto-generated destructor stub
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::alloc_network_learning_rates ()
   {
      if (network_learning_rates == NULL)
         network_learning_rates = new LRPolicy (neural_network);
   }

   template<class EFunc, class LRPolicy>
   BaseNeuralNet &TemporalDiffTrainer3<EFunc, LRPolicy>::get_neuralnet ()
   {
      return neural_network;
   }

   template<class EFunc, class LRPolicy>
   EFunc &TemporalDiffTrainer3<EFunc, LRPolicy>::get_errorfunc ()
   {
      return error_func;
   }

   template<class EFunc, class LRPolicy>
   LRPolicy &TemporalDiffTrainer3<EFunc, LRPolicy>::get_learning_rates ()
   {
      LRPolicy *learning_rates = dynamic_cast<LRPolicy *>(network_learning_rates);
      return *learning_rates;
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::set_lambda (double l)
   {
      lambda = l;
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::set_gamma (double g)
   {
      gamma = g;
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::set_predict_final_cost ()
   {
      predict_mode = FINAL_COST;
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::set_predict_cumulative_cost ()
   {
      predict_mode = CUMULATIVE_COST;
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::train (
      const DataSet<Exemplar<Pattern, Pattern> > &trnset,
      const DataSet<Exemplar<Pattern, Pattern> > &vldset,
      const DataSet<Exemplar<Pattern, Pattern> > &tstset)
   {

   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::train (
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_trnset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_vldset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_tstset)
   {
      int stop_status = TRAINING;
      long epoch_no;
      unsigned int trainset_ndx;

      double perf;
      double prev_global_performance = 1.0e+15;
      double global_performance = 0;
      double prev_valid_perf = DBL_MAX;
      int validation_failures = 0;
      unsigned int consecutive_failback_count = 0;
      bool new_best_flag = false;

      unsigned int permute_ndx;
      vector<unsigned int> presentation_order (_trnset.size ());

      cout << "training set size " << _trnset.size () << endl;

      init_train (_trnset, _vldset, _tstset);
      initialize_presentation_order (presentation_order);

      for (epoch_no = 0; stop_status == TRAINING; epoch_no++)
      {
         global_performance = 0;
         init_training_epoch ();

         if (is_random_order_mode ())
            permute_presentation_order (presentation_order);

         int cnt = 0;
         for (trainset_ndx = 0; trainset_ndx < _trnset.size (); trainset_ndx++)
         {
            cnt++;
            permute_ndx = presentation_order[trainset_ndx];
            pattern_no = trainset_ndx; // ????

            perf = train_exemplar (_trnset.at (permute_ndx));
            global_performance += perf;

            if (is_online_mode ())
            {
               apply_delta_network_weights ();
               zero_delta_network_weights ();
               zero_accumulated_network_gradients ();
            }
         }
         //global_performance /= _trnset.size();

         network_learning_rates->apply_learning_rate_adjustments ();

         if (is_batch_mode ())
            apply_delta_network_weights ();

         if (is_verbose_mode ())
            cout << "global perf(" << epoch_no << ") = " << global_performance
                 << endl;

         // TODO - calculate performance gradient
         // TODO - if we did weight adjustment with rollback then I'd need to do it here

         TrainingRecord::Entry &perf_rec = calc_performance_data (epoch_no + 1,
                                                                  _trnset, _vldset, _tstset);

         // TODO - make the threshold a variable that can be set
         // TODO - make the learning rate reduction a variable that can be set
         // if error increases by 3% then back out weight changes
         //if ((perf_rec.training_perf() - global_performance) / global_performance
         //      > 0.05)
         if ((perf_rec.training_perf () - prev_global_performance) / prev_global_performance
             > 0.05)
         {
            cout << "fail back weights because of large increase in error. "
                 << prev_global_performance << " => " << perf_rec.training_perf ()
                 << endl;

            if ((perf_rec.training_perf () - prev_global_performance)
                / prev_global_performance > 0.5)
            {
               cout << " ========= WHOA! Big Change in Error ===========" << endl;

               cout << "***** FAILBACK WEIGHTS ********" << endl;
               NetworkWeightsData &failback_weights = get_cached_network_weights (
                  failback_weights_id);
               failback_weights.print ();

               cout << "***** UPDATED WEIGHTS ********" << endl;
               NetworkWeightsData trained_weights =
                  neural_network.get_network_weights ();
               trained_weights.print ();

               cout << "***** WEIGHT DELTAS ********" << endl;
               NetworkWeightsData &weight_deltas = get_cached_network_weights (
                  delta_network_weights_id);
               weight_deltas.print ();

               print_accumulated_network_gradients ();
            }

            consecutive_failback_count++;
            if (consecutive_failback_count > 20)
            {
               cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
               update_training_record (epoch_no + 1, MAX_CONSECUTIVE_FAILBACK_STOP,
                                       perf_rec);
               break;
            }

            restore_weights (failback_weights_id);
            network_learning_rates->reduce_learning_rate (0.2);

            epoch_no--;

            continue;
         }
         else
         {
            prev_global_performance = perf_rec.training_perf ();
            consecutive_failback_count = 0;
         }

         if (is_best_perf (perf_rec))
            save_weights (best_weights_id);

         stop_status = is_training_complete (perf_rec, prev_valid_perf,
                                             validation_failures);
         update_training_record (epoch_no + 1, stop_status, perf_rec);
      }

      // Restore weights to the best training epoch
      restore_weights (best_weights_id);
      cout << "terminated training on status " << stop_status << endl;
   }

   template<class EFunc, class LRPolicy>
   double TemporalDiffTrainer3<EFunc, LRPolicy>::train_exemplar (
      const Exemplar<PatternSequence, PatternSequence> &exemplar)
   {
      vector<double> gradient (neural_network.get_output_size ());
      Pattern prev_opatt = vector<double> (neural_network.get_output_size ());
      Pattern opatt = vector<double> (neural_network.get_output_size ());

      PatternSequence ipattseq = exemplar.input ();
      PatternSequence tgtpattseq = exemplar.target_output ();

      NetworkWeightsData &network_deltas = get_cached_network_weights (
         delta_network_weights_id);

      neural_network.clear_error ();
      zero_accumulated_network_gradients ();

      double patt_sse = 0;
      double seq_sse = 0;
      unsigned int pattern_ndx;
      Pattern prev_ipatt;

      vector<double> cum_tgt_vec (neural_network.get_output_size ());
      vector<double> pred_vec (neural_network.get_output_size ());
      Pattern cum_tgtpatt;

      //cout << "********************" << endl;
      for (pattern_ndx = 0; pattern_ndx < ipattseq.size (); pattern_ndx++)
      {
         const Pattern &ipatt = ipattseq.at (pattern_ndx);
         const Pattern &tgtpatt = tgtpattseq.at (pattern_ndx);

         // Present input pattern and get output

         const Pattern &netout = neural_network (ipatt);
         opatt = netout;

         // Save network gradients before accumulating new gradient
         save_network_gradients ();

         accumulate_network_gradients ();

         // The first pattern can't learn anything
         if (pattern_ndx > 0)
         {
            prev_opatt = neural_network (prev_ipatt);

            if (predict_mode == FINAL_COST)
            {
               if (tgtpatt ().size () > 0)
                  error_func (patt_sse, gradient, prev_opatt, tgtpatt);
               else
                  error_func (patt_sse, gradient, prev_opatt, opatt);
            }
            else if (predict_mode == CUMULATIVE_COST)
            {
               if (pattern_ndx < ipattseq.size () - 1)
                  for (unsigned int i = 0; i < opatt ().size (); i++)
                  {
                     //cum_tgt_vec[i] = 0.75 * opatt().at(i) + tgtpatt().at(i);
                     cum_tgt_vec[i] = opatt ().at (i) - tgtpatt ().at (i);
                  }
               else
                  for (unsigned int i = 0; i < opatt ().size (); i++)
                     cum_tgt_vec[i] = tgtpatt ().at (i);

               //pred_vec = prev_opatt;
               for (unsigned int i = 0; i < prev_opatt ().size (); i++)
                  pred_vec[i] = prev_opatt ().at (i);

               //cout << "opatt[0] = " << opatt().at(0) << endl;

               //error_func(patt_sse, gradient, pred_vec, cum_tgt_vec);

               //if (pattern_ndx < ipattseq.size() - 1)
               if (tgtpatt ().at (0) == 0.0)
                  patt_sse = -(tgtpatt ().at (0) - pred_vec[0]);
               else
                  patt_sse = -((gamma * opatt ().at (0) + tgtpatt ().at (0))
                               - pred_vec[0]);

               //else
               //   patt_sse = -(tgtpatt().at(0) - pred_vec[0]);

               gradient[0] = patt_sse;

               patt_sse = 0.5 * (patt_sse * patt_sse);
            }

            seq_sse += patt_sse;

            neural_network.clear_error ();
            //neural_network.backprop(gradient);
            td_backprop (gradient);

            network_learning_rates->update_learning_rate_adjustments (1);

            /*
             * Calculate the weight updates for the PREVIOUS network activation
             */
            const vector<BaseLayer *> network_layers =
               neural_network.get_network_layers ();
            for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
            {
               BaseLayer &layer = *network_layers[ndx];
               const string &name = layer.name ();

               LayerWeightsData &layer_deltas = network_deltas.layer_weights (name);

               calc_layer_bias_adj (layer.get_error (), layer, 1,
                                    layer_deltas.biases);
               calc_layer_weight_adj (layer.get_error (), layer, 1,
                                      layer_deltas.weights);
            }
         }

         // Save current input
         prev_ipatt = ipatt;

         /*
          for (unsigned int i = 0; i < prev_outputv.size(); i++)
          prev_outputv[i] = netout[i];
          */
      } /* loop through patterns */

      //seq_sse = (ipattseq.size() > 0) ? seq_sse / ipattseq.size() : 0;

      return seq_sse;
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::save_network_gradients ()
   {
      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         string name = layer.name ();

         // ***** Save cumulative dAdB
         Array<double> &cumulative_dAdB = cumulative_dAdB_map[name];
         Array<double> &prev_cumulative_dAdB = prev_cumulative_dAdB_map[name];

         prev_cumulative_dAdB = cumulative_dAdB;

         // ***** Save cumulative dAdN
         Array<double> &cumulative_dAdN = cumulative_dAdN_map[name];
         Array<double> &prev_cumulative_dAdN = prev_cumulative_dAdN_map[name];

         prev_cumulative_dAdN = cumulative_dAdN;

         // ***** Save cumulative dNdW
         Array<double> &cumulative_dNdW = cumulative_dNdW_map[name];
         Array<double> &prev_cumulative_dNdW = prev_cumulative_dNdW_map[name];

         prev_cumulative_dNdW = cumulative_dNdW;

         // ***** Save cumulative dAdW
         Array<double> &cumulative_dAdW = cumulative_dAdW_map[name];
         Array<double> &prev_cumulative_dAdW = prev_cumulative_dAdW_map[name];

         prev_cumulative_dAdW = cumulative_dAdW;

         // ***** Save cumulative dNdI
         Array<double> &cumulative_dNdI = cumulative_dNdI_map[name];
         Array<double> &prev_cumulative_dNdI = prev_cumulative_dNdI_map[name];

         prev_cumulative_dNdI = cumulative_dNdI;
      }
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::accumulate_network_gradients ()
   {
      //vector<double> unity( neural_network.get_output_size(), 1.0 );
      //neural_network.clear_error();
      //neural_network.backprop(unity);

      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         string name = layer.name ();

         double prev_sum;
         double temp = 0;

         unsigned int layer_size = layer.size ();
         unsigned int layer_input_size = layer.input_size ();

         const vector<double> &errorv = layer.get_error ();

         /*
          * Because we backpropagated a unity vector, dEdB is the gradient, dAdB
          */
         Array<double> &cumulative_dAdB = cumulative_dAdB_map[name];
         const Array<double> &dAdB = layer.get_dAdB ();

         for (unsigned int out_ndx = 0; out_ndx < cumulative_dAdB.rowDim ();
              out_ndx++)
         {
            for (unsigned int bias_ndx = 0; bias_ndx < cumulative_dAdB.colDim ();
                 bias_ndx++)
            {
               prev_sum = cumulative_dAdB.at (out_ndx, bias_ndx);
               cumulative_dAdB.at (out_ndx, bias_ndx) = dAdB.at (out_ndx, bias_ndx)
                                                        + lambda * prev_sum;
            }
         }

         Array<double> &cumulative_dAdN = cumulative_dAdN_map[name];
         const Array<double> &dAdN = layer.get_dAdN ();

         for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
         {
            for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
            {
               prev_sum = cumulative_dAdN.at (out_ndx, netin_ndx);
               cumulative_dAdN.at (out_ndx, netin_ndx) = dAdN.at (out_ndx, netin_ndx)
                                                         + lambda * prev_sum;
            }
         }

         Array<double> &cumulative_dNdW = cumulative_dNdW_map[name];
         const Array<double> &dNdW = layer.get_dNdW ();

         for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
         {
            for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
            {
               prev_sum = cumulative_dNdW.at (netin_ndx, in_ndx);
               cumulative_dNdW.at (netin_ndx, in_ndx) = dNdW.at (netin_ndx, in_ndx)
                                                        + lambda * prev_sum;
            }
         }

         /*
          * Because we backpropagated a unity vector, dEdW is the gradient, dAdW
          */
         Array<double> &cumulative_dAdW = cumulative_dAdW_map[name];
         const Array<double> &dEdW = layer.get_dEdW ();

         for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
         {
            for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
            {
               prev_sum = cumulative_dAdW.at (netin_ndx, in_ndx);
               cumulative_dAdW.at (netin_ndx, in_ndx) = dEdW.at (netin_ndx, in_ndx)
                                                        + lambda * prev_sum;
            }
         }

         Array<double> &cumulative_dNdI = cumulative_dNdI_map[name];
         const Array<double> &dNdI = layer.get_dNdI ();

         for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
         {
            for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
            {
               prev_sum = cumulative_dNdI.at (netin_ndx, in_ndx);
               cumulative_dNdI.at (netin_ndx, in_ndx) = dNdI.at (netin_ndx, in_ndx)
                                                        + lambda * prev_sum;
            }
         }
      }
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::td_backprop (
      const vector<double> &_errorv)
   {
      unsigned int timeStep = 1;
      /*
       * Backprop network error to output layers
       */
      ConnectionMap &network_output_map =
         neural_network.get_network_output_connection_map ();
      const vector<vector<double> > &network_errorv = network_output_map.get_error (
         _errorv);

      const vector<ConnectionEntry> &network_output_connvec =
         network_output_map.get_input_connections ();
      for (int conn_ndx = 0; conn_ndx < network_output_connvec.size (); conn_ndx++)
      {
         const ConnectionEntry &conn = network_output_connvec.at (conn_ndx);
         BaseLayer &layer = conn.get_input_layer ();

         layer.backprop (network_errorv.at (conn_ndx));
      }

      /*
       * Backprop error through network in reverse order of the layer activation ordering
       */
      vector<BaseLayer *> layer_activation_order =
         neural_network.get_layer_activation_order ();
      for (int i = layer_activation_order.size () - 1; i >= 0; i--)
      {
         BaseLayer &layer = *layer_activation_order[i];

         /*
          * Manually backprop throught the layer using the cumulative gradients
          */
         Array<double> &cumulative_dAdN = prev_cumulative_dAdN_map[layer.name ()];
         Array<double> &cumulative_dNdI = prev_cumulative_dNdI_map[layer.name ()];

         unsigned int layer_size = layer.size ();
         unsigned int input_layer_size = layer.input_size ();

         vector<double> temp_layer_input_error (input_layer_size);

         const vector<double> &layer_errorv = layer.get_error (timeStep);
         for (unsigned int in_ndx = 0; in_ndx < input_layer_size; in_ndx++)
         {
            temp_layer_input_error.at (in_ndx) = 0;

            for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
            {
               /*
                layer_input_error.at(timeStep).at(in_ndx) += errorv.at(out_ndx)
                * dAdI.at(timeStep).at(out_ndx, in_ndx);
                */

               for (unsigned int netin_ndx = 0; netin_ndx < layer_size;
                    netin_ndx++)
                  temp_layer_input_error.at (in_ndx) += layer_errorv.at (out_ndx)
                                                        * cumulative_dAdN.at (out_ndx, netin_ndx)
                                                        * cumulative_dNdI.at (netin_ndx, in_ndx);
            }
         }

         // backprop error at the inputs to layers providing input to this layer
         backprop_scatter (temp_layer_input_error, layer, 1);
      }

   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::backprop_scatter (
      const vector<double> &ierrorv, BaseLayer &layer, unsigned int timeStep)
   {
      /*
       * This scatters the consolidated error vector to the individual
       * error vectors for all layers feeding this one
       */
      ConnectionMap *layer_input = neural_network.get_layer_connection_map (layer);
      const vector<vector<double> > &scattered_errorv = layer_input->get_error (
         ierrorv);

      const vector<ConnectionEntry> &connvec =
         layer_input->get_input_connections ();

      vector<vector<double> > &network_input_error =
         neural_network.get_input_error ();

      for (unsigned int conn_ndx = 0; conn_ndx < connvec.size (); conn_ndx++)
      {
         const ConnectionEntry &conn = connvec.at (conn_ndx);

         if (conn.is_input_connection ())
         {
            unsigned int ipatt_ndx = conn.get_input_pattern_index ();
            vector<double> &errvec = network_input_error.at (ipatt_ndx);

            for (unsigned int vec_ndx = 0; vec_ndx < errvec.size (); vec_ndx++)
               errvec.at (vec_ndx) += scattered_errorv.at (conn_ndx).at (vec_ndx);
            continue;
         }

         unsigned int bpTimeStep = timeStep;

         BaseLayer &layer = conn.get_input_layer ();
         layer.backprop (scattered_errorv.at (conn_ndx), bpTimeStep);
      }
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::calc_layer_bias_adj (
      const vector<double> &errorv, const BaseLayer &layer,
      unsigned int timeStep, vector<double> &biasDelta)
   {
      Array<double> &cumulative_dAdB = prev_cumulative_dAdB_map[layer.name ()];

      // Get the learning rates for the biases
      vector<double> layer_bias_lr =
         network_learning_rates->get_bias_learning_rates ().at (layer.name ());

      for (unsigned int bias_ndx = 0; bias_ndx < biasDelta.size (); bias_ndx++)
         for (unsigned int out_ndx = 0; out_ndx < layer.size (); out_ndx++)
            biasDelta.at (bias_ndx) += -layer_bias_lr[bias_ndx] * errorv.at (out_ndx)
                                       * cumulative_dAdB.at (out_ndx, bias_ndx);
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::calc_layer_weight_adj (
      const vector<double> &errorv, const BaseLayer &layer,
      unsigned int timeStep, Array<double> &weightDelta)
   {
      Array<double> &cumulative_dAdN = prev_cumulative_dAdN_map[layer.name ()];
      Array<double> &cumulative_dNdW = prev_cumulative_dNdW_map[layer.name ()];
      Array<double> &cumulative_dAdW = prev_cumulative_dAdW_map[layer.name ()];

      // Get the learning rates for the weights
      Array<double> layer_weights_lr =
         network_learning_rates->get_weight_learning_rates ().at (layer.name ());

      unsigned int layer_size = layer.size ();
      unsigned int layer_input_size = layer.input_size ();

      double temp;
      for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
      {
         temp = 0;
         for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
            temp += errorv.at (out_ndx) * cumulative_dAdN.at (out_ndx, netin_ndx);

         for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
            weightDelta.at (netin_ndx, in_ndx) += -layer_weights_lr.at (netin_ndx,
                                                                        in_ndx) * temp
                                                  * cumulative_dNdW.at (netin_ndx, in_ndx);
      }
   }

/*
 * Instantiate and initialize any data structures needed to train the network as required.
 * For example structures to hold and accumulate the weight and bias deltas.
 */
   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::init_train (
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_trnset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_vldset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_tstset)
   {
      neural_network.clear_error ();
      zero_delta_network_weights ();

      network_learning_rates->reset ();

      TrainingRecord::Entry &entry = calc_performance_data (0, _trnset, _vldset,
                                                            _tstset);

      clear_training_record ();
      unsigned int epoch = 0;
      unsigned int stop_sig = BaseTrainer::TRAINING;
      update_training_record (epoch, stop_sig, entry);

      save_weights (failback_weights_id);
      save_weights (best_weights_id);

      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         cumulative_dAdB_map[name] = Array<double> ();
         cumulative_dAdB_map[name].resize (output_sz, output_sz);

         cumulative_dAdN_map[name] = Array<double> ();
         cumulative_dAdN_map[name].resize (output_sz, output_sz);

         cumulative_dNdW_map[name] = Array<double> ();
         cumulative_dNdW_map[name].resize (output_sz, input_sz);

         cumulative_dNdI_map[name] = Array<double> ();
         cumulative_dNdI_map[name].resize (output_sz, input_sz);

         cumulative_dAdW_map[name] = Array<double> ();
         cumulative_dAdW_map[name].resize (output_sz, input_sz);

         prev_cumulative_dAdB_map[name] = Array<double> ();
         prev_cumulative_dAdB_map[name].resize (output_sz, output_sz);

         prev_cumulative_dAdN_map[name] = Array<double> ();
         prev_cumulative_dAdN_map[name].resize (output_sz, output_sz);

         prev_cumulative_dNdW_map[name] = Array<double> ();
         prev_cumulative_dNdW_map[name].resize (output_sz, input_sz);

         prev_cumulative_dNdI_map[name] = Array<double> ();
         prev_cumulative_dNdI_map[name].resize (output_sz, input_sz);

         prev_cumulative_dAdW_map[name] = Array<double> ();
         prev_cumulative_dAdW_map[name].resize (output_sz, input_sz);
      }

      //   prev_outputv.resize(neural_network.get_output_size());
   }

/*
 * Perform any initialization required for the new training epoch. For example clear
 * all data structures required to accumulate the new global network error, the weight
 * and bias deltas and etc.
 */
   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::init_training_epoch ()
   {
      neural_network.clear_error ();
      zero_delta_network_weights ();
      zero_accumulated_network_gradients ();
      save_weights (failback_weights_id);
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::zero_accumulated_network_gradients ()
   {
      /* ******************************************
       *    Zero bias and weight deltas
       */
      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         Array<double> &cumulative_dAdB = cumulative_dAdB_map[name];
         Array<double> &cumulative_dAdN = cumulative_dAdN_map[name];
         Array<double> &cumulative_dNdW = cumulative_dNdW_map[name];
         Array<double> &cumulative_dNdI = cumulative_dNdI_map[name];
         Array<double> &cumulative_dAdW = cumulative_dAdW_map[name];

         cumulative_dAdB = 0;
         cumulative_dAdN = 0;
         cumulative_dNdW = 0;
         cumulative_dNdI = 0;
         cumulative_dAdW = 0;
      }
   }

   template<class EFunc, class LRPolicy>
   void TemporalDiffTrainer3<EFunc, LRPolicy>::print_accumulated_network_gradients ()
   {
      /* ******************************************
       *    Zero bias and weight deltas
       */
      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         Array<double> &cumulative_dAdB = cumulative_dAdB_map[name];
         Array<double> &cumulative_dAdN = cumulative_dAdN_map[name];
         Array<double> &cumulative_dNdW = cumulative_dNdW_map[name];
         Array<double> layer_weights_lr =
            network_learning_rates->get_weight_learning_rates ().at (name);

         cout << "**** cumulative dAdN *****" << endl;
         for (unsigned int i = 0; i < cumulative_dAdN.rowDim (); i++)
         {
            for (unsigned int j = 0; j < cumulative_dAdN.colDim (); j++)
            {
               cout << cumulative_dAdN.at (i, j) << " ";
            }
            cout << endl;
         }
         cout << "************" << endl;

         cout << "**** cumulative dNdW *****" << endl;
         for (unsigned int i = 0; i < cumulative_dNdW.rowDim (); i++)
         {
            for (unsigned int j = 0; j < cumulative_dNdW.colDim (); j++)
            {
               cout << cumulative_dNdW.at (i, j) << " ";
            }
            cout << endl;
         }
         cout << "************" << endl;

         cout << "**** learning rates *****" << endl;
         for (unsigned int i = 0; i < layer_weights_lr.rowDim (); i++)
         {
            for (unsigned int j = 0; j < layer_weights_lr.colDim (); j++)
            {
               cout << layer_weights_lr.at (i, j) << " ";
            }
            cout << endl;
         }
         cout << "************" << endl;
      }
   }

   template<class EFunc, class LRPolicy>
   TrainingRecord::Entry &TemporalDiffTrainer3<EFunc, LRPolicy>::calc_performance_data (
      unsigned int _epoch,
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_trnset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_vldset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_tstset)
   {
      static TrainingRecord::Entry entry (_epoch);
      double perf;

      entry.set_epoch (_epoch);

      perf = sim (_trnset);
      entry.set_training_perf (perf);

      if (_vldset.size () > 0)
      {
         perf = sim (_vldset);
         entry.set_validation_perf (perf);
      }

      if (_tstset.size () > 0)
      {
         perf = sim (_tstset);
         entry.set_test_perf (perf);
      }

      return entry;
   }

   template<class EFunc, class LRPolicy>
   double TemporalDiffTrainer3<EFunc, LRPolicy>::sim (
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_dataset)
   {
      long epoch_no;
      unsigned int trainset_ndx;
      double perf;

      vector<double> gradient (neural_network.get_output_size ());
      Pattern prev_opatt = vector<double> (neural_network.get_output_size ());

      vector<double> cum_tgt_vec (neural_network.get_output_size ());
      vector<double> pred_vec (neural_network.get_output_size ());
      Pattern cum_tgtpatt;

      double global_performance = 0;
      vector<double> est_cost_to_go;
      vector<double> cost_to_go;
      vector<double> diff;

      for (trainset_ndx = 0; trainset_ndx < _dataset.size (); trainset_ndx++)
      {
         cost_to_go.clear ();
         est_cost_to_go.clear ();

         const Exemplar<PatternSequence, PatternSequence> &exemplar = _dataset.at (
            trainset_ndx);

         PatternSequence ipattseq = exemplar.input ();
         PatternSequence tgtpattseq = exemplar.target_output ();

         double patt_sse = 0;
         double seq_sse = 0;
         for (unsigned int pattern_ndx = 0; pattern_ndx < ipattseq.size ();
              pattern_ndx++)
         {
            const Pattern &ipatt = ipattseq.at (pattern_ndx);
            const Pattern &tgtpatt = tgtpattseq.at (pattern_ndx);

            // Present input pattern and get output
            const Pattern &opatt = neural_network (ipatt);

            cost_to_go.push_back (tgtpatt ().at (0));
            est_cost_to_go.push_back (opatt ().at (0));

            if (pattern_ndx == 0)
            {
               prev_opatt = opatt;
               continue;
            }

            // Calculate the output error
            if (predict_mode == FINAL_COST)
            {
               if (tgtpatt ().size () > 0)
                  error_func (patt_sse, gradient, prev_opatt, tgtpatt);
               else
                  error_func (patt_sse, gradient, prev_opatt, opatt);
            }
            else if (predict_mode == CUMULATIVE_COST)
            {
               if (pattern_ndx < ipattseq.size () - 1)
                  for (unsigned int i = 0; i < opatt ().size (); i++)
                     cum_tgt_vec[i] = gamma * opatt ().at (i) + tgtpatt ().at (i);
               else
                  for (unsigned int i = 0; i < opatt ().size (); i++)
                     cum_tgt_vec[i] = tgtpatt ().at (i);

               //pred_vec = prev_opatt;
               for (unsigned int i = 0; i < prev_opatt ().size (); i++)
                  pred_vec[i] = prev_opatt ().at (i);

               error_func (patt_sse, gradient, prev_opatt, cum_tgt_vec);

               for (unsigned int i = 0; i < prev_opatt ().size (); i++)
                  pred_vec[i] = prev_opatt ().at (i);

               //cout << "opatt[0] = " << opatt().at(0) << endl;

               //error_func(patt_sse, gradient, pred_vec, cum_tgt_vec);
               //if (pattern_ndx < ipattseq.size() - 1)
               if (tgtpatt ().at (0) == 0.0)
                  patt_sse = -(tgtpatt ().at (0) - pred_vec[0]);
               else
                  patt_sse = -((gamma * opatt ().at (0) + tgtpatt ().at (0))
                               - pred_vec[0]);
               //else
               //   -(patt_sse = tgtpatt().at(0) - pred_vec[0]);

               patt_sse = 0.5 * (patt_sse * patt_sse);
            }

            if (pattern_ndx > 0)
               seq_sse += patt_sse;

            prev_opatt = opatt;
         }
         //seq_sse = (ipattseq.size() > 0) ? seq_sse / ipattseq.size() : 0.0;

         /*
         // IGNORE THE CUMULATIVE COST SECTION ABOVE FOR THE MOMENT
         if (predict_mode == CUMULATIVE_COST)
         {
            seq_sse = 0;
            if (cost_to_go.size() > 0)
            {
               diff.resize(cost_to_go.size());

               patt_sse = cost_to_go.at(cost_to_go.size()-1) - est_cost_to_go.at(cost_to_go.size()-1);
               diff.at(cost_to_go.size()-1) = patt_sse * patt_sse;
            }

            for (unsigned int i = cost_to_go.size() - 1; i > 0; i--)
            {
               est_cost_to_go.at(i - 1) += est_cost_to_go.at(i);
               cost_to_go.at(i - 1) += cost_to_go.at(i);

               patt_sse = cost_to_go.at(i - 1) - est_cost_to_go.at(i - 1);
               diff.at(cost_to_go.size()-1) = patt_sse * patt_sse;
            }

            seq_sse = 0;
            double x;

            for (int i = cost_to_go.size()-1; i > 0; i--)
            {
               if (i < cost_to_go.size()-1)
                  patt_sse = (gamma * est_cost_to_go.at(i)  + (cost_to_go.at(i) - cost_to_go.at(i-1))) - est_cost_to_go.at(i-1);
               else
                  patt_sse = (cost_to_go.at(i) - cost_to_go.at(i-1)) - est_cost_to_go.at(i-1);

               seq_sse = patt_sse*patt_sse;
               //seq_sse = patt_sse*patt_sse + lambda * seq_sse;
            }

            seq_sse /= ipattseq.size();
         }
         */

         global_performance += seq_sse;
      }
      //global_performance /= _dataset.size();

      return global_performance;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_TEMPDIFFTRAINER3_H_ */
