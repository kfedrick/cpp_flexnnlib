/*
 * TDCTrainer22.h
 *
 *  Created on: Feb 26, 2014
 *      Author: kfedrick
 *
 *  Temporal difference with gradient correction trainer
 *  Sutton 2009 formula.
 */

#ifndef FLEX_NEURALNET_TDCTRAINER2_H_
#define FLEX_NEURALNET_TDCTRAINER2_H_

#include "TDCNeuralNet.h"
#include "BaseTrainer.h"
#include <cfloat>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace flexnnet
{

   template<class EFunc, class LRPolicy>
   class TDCTrainer2 : public BaseTrainer
   {
   public:
      TDCTrainer2 (TDCNeuralNet &_net);
      virtual ~TDCTrainer2 ();

      void alloc_network_learning_rates ();

      void set_slow_learning_rate_multiplier (double _val);
      LRPolicy &get_learning_rates ();

      void set_predict_final_cost ();
      void set_predict_cumulative_cost ();

      void set_lambda (double l);
      void set_gamma (double l);

      void set_print_gradient (bool _val);

      void save_gradient (map<string, vector<double> > &_dEdB_map, map<string, Array<double> > &_dEdW_map, map<string,
                                                                                                               Array<
                                                                                                                  double>> &_Hv_map);

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

      bool
      InvertMatrix (const boost::numeric::ublas::matrix<double> &input, boost::numeric::ublas::matrix<double> &inverse);

      double sim2 (
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

      TDCNeuralNet &get_neuralnet ();
      EFunc &get_errorfunc ();

   private:

      void calc_layer_weight_adj (double _tdErr, NetworkWeightsData &_nnDeltas);

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
      TDCNeuralNet &neural_network;

      /* ***********************************************
       *    Learning rate parameters
       */
      double slow_lr_multiplier;
      double lambda;
      double gamma;

      Prediction_Mode predict_mode;

      map<string, vector<double> > prev_dEdB_map;
      map<string, Array<double> > prev_dEdW_map;
      map<string, Array<double> > prev_Hv_map;

      map<string, Array<double> > w_map;

      map<string, Array<double>> gE_td_phi;

      unsigned int pattern_no;

      bool print_gradient;
   };

   template<class EFunc, class LRPolicy>
   TDCTrainer2<EFunc, LRPolicy>::TDCTrainer2 (TDCNeuralNet &_net) :
      neural_network (_net), BaseTrainer (_net)
   {
      set_batch_mode ();
      set_predict_final_cost ();
      set_random_order_mode (true);
      set_verbose (false);
      set_global_learning_rate (0.05);
      set_slow_learning_rate_multiplier (2.0);
      set_lambda (0.0);

      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         prev_dEdB_map[name] = vector<double> ();
         prev_dEdB_map[name].resize (output_sz);

         prev_dEdW_map[name] = Array<double> ();
         prev_dEdW_map[name].resize (output_sz, input_sz);

         w_map[name] = Array<double> ();
         w_map[name].resize (output_sz, input_sz + 1);
      }
   }

   template<class EFunc, class LRPolicy>
   TDCTrainer2<EFunc, LRPolicy>::~TDCTrainer2 ()
   {
      // TODO Auto-generated destructor stub
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::alloc_network_learning_rates ()
   {
      if (network_learning_rates == NULL)
         network_learning_rates = new LRPolicy (neural_network);
   }

   template<class EFunc, class LRPolicy>
   TDCNeuralNet &TDCTrainer2<EFunc, LRPolicy>::get_neuralnet ()
   {
      return neural_network;
   }

   template<class EFunc, class LRPolicy>
   EFunc &TDCTrainer2<EFunc, LRPolicy>::get_errorfunc ()
   {
      return error_func;
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::set_slow_learning_rate_multiplier (double _val)
   {
      slow_lr_multiplier = _val;
   }

   template<class EFunc, class LRPolicy>
   LRPolicy &TDCTrainer2<EFunc, LRPolicy>::get_learning_rates ()
   {
      LRPolicy *learning_rates = dynamic_cast<LRPolicy *>(network_learning_rates);
      return *learning_rates;
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::set_lambda (double l)
   {
      lambda = l;
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::set_gamma (double g)
   {
      gamma = g;
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::set_predict_final_cost ()
   {
      predict_mode = FINAL_COST;
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::set_predict_cumulative_cost ()
   {
      predict_mode = CUMULATIVE_COST;
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::set_print_gradient (bool _val)
   {
      print_gradient = _val;
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::save_gradient (map<string, vector<double> > &_dEdB_map, map<string,
                                                                                                  Array<double> > &_dEdW_map, map<
      string,
      Array<double>> &_Hv_map)
   {
      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         _dEdB_map[name] = layer.get_dEdB ();
         _dEdW_map[name] = layer.get_dEdW ();
         _Hv_map[name] = neural_network.get_Hv (name);
      }
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::train (
      const DataSet<Exemplar<Pattern, Pattern> > &trnset,
      const DataSet<Exemplar<Pattern, Pattern> > &vldset,
      const DataSet<Exemplar<Pattern, Pattern> > &tstset)
   {

   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::train (
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

      unsigned int norm_size = 0;
      unsigned int insize;

      for (trainset_ndx = 0; trainset_ndx < _trnset.size (); trainset_ndx++)
      {
         const Exemplar<PatternSequence, PatternSequence> &exemplar = _trnset.at (trainset_ndx);
         PatternSequence ipattseq = exemplar.input ();
         const Pattern &ipatt = ipattseq.at (0);

         norm_size += ipattseq.size ();

         insize = ipatt ().size ();
      }

      int normfac = 0;
      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         unsigned int osize = layer.size ();
         unsigned int isize = layer.get_input_error ().size ();

         normfac += osize * (isize + 1);
      }

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
            }
         }
         global_performance /= _trnset.size ();

         //network_learning_rates->apply_learning_rate_adjustments();

         if (is_batch_mode ())
         {
            apply_delta_network_weights ();
            zero_delta_network_weights ();

            perf = 0;

            const vector<BaseLayer *> network_layers =
               neural_network.get_network_layers ();
            for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
            {
               BaseLayer &layer = *network_layers[ndx];
               const string &name = layer.name ();

               unsigned int osize = layer.size ();
               unsigned int isize = layer.get_input_error ().size ();
               const Array<double> &w = w_map[name];

               for (unsigned int i = 0; i < osize; i++)
                  for (unsigned int j = 0; j < isize + 1; j++)
                     perf += gE_td_phi[name].at (i, j) / norm_size * w.at (i, j);

               gE_td_phi[name] = 0;
            }

            perf = sqrt (perf);

         }

         if (is_verbose_mode ())
         {
            cout << "perf (" << epoch_no << ") : " << perf << endl;
            //cout << "global perf(" << epoch_no << ") " << global_performance
            //      << endl;
         }

         // TODO - calculate performance gradient
         // TODO - if we did weight adjustment with rollback then I'd need to do it here

         TrainingRecord::Entry &perf_rec = calc_performance_data (epoch_no + 1,
                                                                  _trnset, _vldset, _tstset);

         // TODO - make the threshold a variable that can be set
         // TODO - make the learning rate reduction a variable that can be set
         // if error increases by 3% then back out weight changes
         if ((perf_rec.training_perf () - prev_global_performance)
             / prev_global_performance > 0.05)
         {
            cout << "fail back weights because of large increase in error. "
                 << prev_global_performance << " => " << perf_rec.training_perf ()
                 << endl;

            consecutive_failback_count++;
            if (consecutive_failback_count > 10)
            {
               cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
               update_training_record (epoch_no + 1, MAX_CONSECUTIVE_FAILBACK_STOP,
                                       perf_rec);
               break;
            }

            restore_weights (failback_weights_id);
            network_learning_rates->reduce_learning_rate (0.4);
            epoch_no--;

            continue;
         }
         else
         {
            prev_global_performance = perf_rec.training_perf ();
            consecutive_failback_count = 0;
         }

         if (is_best_perf (perf_rec))
         {
            //cout << "save as best weights" << endl;
            save_weights (best_weights_id);
         }

         stop_status = is_training_complete (perf_rec, prev_valid_perf,
                                             validation_failures);
         update_training_record (epoch_no + 1, stop_status, perf_rec);
      }

      // Restore weights to the best training epoch
      //restore_weights(best_weights_id);
      cout << "terminated training on status " << stop_status << endl;
   }

   template<class EFunc, class LRPolicy>
   double TDCTrainer2<EFunc, LRPolicy>::train_exemplar (
      const Exemplar<PatternSequence, PatternSequence> &exemplar)
   {
      vector<double> egradient (neural_network.get_output_size ());
      vector<double> ugradient (neural_network.get_output_size (), 1.0);

      Pattern prev_opatt = vector<double> (neural_network.get_output_size ());
      Pattern opatt = vector<double> (neural_network.get_output_size ());

      PatternSequence ipattseq = exemplar.input ();
      PatternSequence tgtpattseq = exemplar.target_output ();

      NetworkWeightsData &network_deltas = get_cached_network_weights (
         delta_network_weights_id);

      neural_network.clear_error ();

      double td_error;
      double patt_err = 0;
      double prev_patt_err = 0;

      double patt_sse = 0;
      double seq_sse = 0;
      unsigned int pattern_ndx;
      Pattern prev_ipatt;

      unsigned int insize = 0;
      vector<vector<double> > inerr = neural_network.get_input_error ();
      for (unsigned int i = 0; i < inerr.size (); i++)
         insize += inerr.at (i).size ();

      for (pattern_ndx = 0; pattern_ndx < ipattseq.size (); pattern_ndx++)
      {
         const Pattern &ipatt = ipattseq.at (pattern_ndx);
         const Pattern &tgtpatt = tgtpattseq.at (pattern_ndx);

         save_gradient (prev_dEdB_map, prev_dEdW_map, prev_Hv_map);

         // Present input pattern and get output
         const Pattern &netout = neural_network (ipatt);
         opatt = netout;

         if (print_gradient)
         {
            cout << "***** initial activation *****" << endl;
            for (unsigned int i = 0; i < ipatt ().size (); i++)
               cout << ipatt ().at (i) << " ";
            cout << " => ";
            cout << "critic rsig " << opatt ().at (0) << endl;
         }

         // Save network gradients before accumulating new gradient
         neural_network.clear_error ();
         neural_network.backprop (ugradient);

         // The first pattern can't learn anything
         if (pattern_ndx == 0)
         {
            // Try priming patt_err
            //patt_err = gamma * opatt().at(0);
         }
         else
         {
            if (pattern_ndx < ipattseq.size () - 1)
               td_error = tgtpatt ().at (0) + gamma * opatt ().at (0)
                          - prev_opatt ().at (0);
            else

               td_error = tgtpatt ().at (0) - prev_opatt ().at (0);

            seq_sse += patt_sse;

            /*
             * Calculate the weight updates for the PREVIOUS network activation
             */
            calc_layer_weight_adj (td_error, network_deltas);

            const vector<BaseLayer *> network_layers =
               neural_network.get_network_layers ();
            for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
            {
               BaseLayer &layer = *network_layers[ndx];
               const string &name = layer.name ();

               unsigned int osize = layer.size ();
               unsigned int isize = layer.get_input_error ().size ();

               const Array<double> &prev_dEdW = prev_dEdW_map[name];
               const Array<double> &w = w_map[name];
               neural_network.set_v (name, w);

               for (unsigned int i = 0; i < osize; i++)
               {
                  gE_td_phi[name].at (i, 0) += td_error;
                  for (unsigned int j = 0; j < isize; j++)
                     gE_td_phi[name].at (i, j + 1) += td_error * prev_dEdW.at (i, j);
               }
            }
         }

         // Save current input
         prev_ipatt = ipatt;
         prev_opatt = opatt;

      } /* loop through patterns */



      seq_sse = (ipattseq.size () > 0) ? seq_sse / ipattseq.size () : 0;

      return seq_sse;
   }

   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::calc_layer_weight_adj (double _tdErr, NetworkWeightsData &_nnDeltas)
   {
      /*
       * Calculate phi * w
       */
      double phi_w = 0;

      const vector<BaseLayer *> network_layers =
         neural_network.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         const vector<double> &prev_dEdB = prev_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_dEdW_map[name];
         const Array<double> &w = w_map[name];

         for (unsigned int rowndx = 0; rowndx < prev_dEdW.rowDim (); rowndx++)
         {
            phi_w += prev_dEdB.at (rowndx) * w.at (rowndx, 0);
            for (unsigned int colndx = 0; colndx < prev_dEdW.colDim (); colndx++)
               phi_w += prev_dEdW.at (rowndx, colndx) * w.at (rowndx, colndx + 1);
         }
      }

      double h;

      double hb;
      double theta_p_b;
      double theta_p;

      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int osize = layer.size ();
         unsigned int isize = layer.input_size ();

         // Get the learning rates for the biases
         vector<double> layer_bias_lr =
            network_learning_rates->get_bias_learning_rates ().at (layer.name ());

         // Get the learning rates for the weights
         Array<double> layer_weights_lr =
            network_learning_rates->get_weight_learning_rates ().at (layer.name ());

         const Array<double> &Hv = prev_Hv_map.at (layer.name ());

         const vector<double> &dEdB = layer.get_dEdB ();
         const Array<double> &dEdW = layer.get_dEdW ();

         const vector<double> &prev_dEdB = prev_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_dEdW_map[name];
         const Array<double> &w = w_map[name];

         LayerWeightsData &layer_deltas = _nnDeltas.layer_weights (name);

         for (unsigned int oNdx = 0; oNdx < osize; oNdx++)
         {
            for (unsigned int iNdx = 0; iNdx < isize; iNdx++)
            {

               h = (_tdErr - phi_w) * Hv.at (oNdx, iNdx);

               theta_p = _tdErr * prev_dEdW.at (oNdx, iNdx)
                         - gamma * dEdW.at (oNdx, iNdx) * phi_w - h;

               layer_deltas.weights.at (oNdx, iNdx) += layer_weights_lr.at (oNdx, iNdx)
                                                       * theta_p;
            }

            hb = (_tdErr - phi_w) * Hv.at (oNdx, 0);
            theta_p_b = _tdErr * prev_dEdB.at (oNdx) - gamma * dEdB.at (oNdx) * phi_w - hb;
            layer_deltas.biases.at (oNdx) += layer_bias_lr.at (oNdx) * theta_p_b;
         }
      } // end loop through layers


      /*
       * Update w estimate
       */
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int osize = layer.size ();
         unsigned int isize = layer.input_size ();

         // Get the learning rates for the biases
         vector<double> layer_bias_lr =
            network_learning_rates->get_bias_learning_rates ().at (layer.name ());

         // Get the learning rates for the weights
         Array<double> layer_weights_lr =
            network_learning_rates->get_weight_learning_rates ().at (layer.name ());

         const vector<double> &prev_dEdB = prev_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_dEdW_map[name];
         Array<double> &w = w_map[name];

         for (unsigned int oNdx = 0; oNdx < osize; oNdx++)
         {
            w.at (oNdx, 0) = w.at (oNdx, 0)
                             + slow_lr_multiplier * layer_bias_lr.at (oNdx) * (_tdErr - phi_w)
                               * prev_dEdB.at (oNdx);

            for (unsigned int iNdx = 0; iNdx < isize; iNdx++)
            {
               w.at (oNdx, iNdx + 1) = w.at (oNdx, iNdx + 1)
                                       + slow_lr_multiplier * layer_weights_lr.at (oNdx, iNdx)
                                         * (_tdErr - phi_w) * prev_dEdW.at (oNdx, iNdx);
            }
         }
      } // end loop through layers
   }

/*
 * Instantiate and initialize any data structures needed to train the network as required.
 * For example structures to hold and accumulate the weight and bias deltas.
 */
   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::init_train (
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

         prev_Hv_map[name].resize (output_sz, input_sz + 1);
         prev_Hv_map[name] = 0;

         prev_dEdB_map[name].assign (prev_dEdB_map[name].size (), 0.0);
         prev_dEdW_map[name] = 0;
         w_map[name] = 0;

         gE_td_phi[name].resize (output_sz, input_sz + 1);
         gE_td_phi[name] = 0;
      }
   }

/*
 * Perform any initialization required for the new training epoch. For example clear
 * all data structures required to accumulate the new global network error, the weight
 * and bias deltas and etc.
 */
   template<class EFunc, class LRPolicy>
   void TDCTrainer2<EFunc, LRPolicy>::init_training_epoch ()
   {
      neural_network.clear_error ();
      zero_delta_network_weights ();
      save_weights (failback_weights_id);

   }

   template<class EFunc, class LRPolicy>
   TrainingRecord::Entry &TDCTrainer2<EFunc, LRPolicy>::calc_performance_data (
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
   double TDCTrainer2<EFunc, LRPolicy>::sim (
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_dataset)
   {
      long epoch_no;
      unsigned int trainset_ndx;
      double perf;
      unsigned int sample_count = 0;

      vector<double> gradient (neural_network.get_output_size ());
      Pattern prev_opatt = vector<double> (neural_network.get_output_size ());

      vector<double> cum_tgt_vec (neural_network.get_output_size ());
      vector<double> pred_vec (neural_network.get_output_size ());
      Pattern cum_tgtpatt;

      double global_performance = 0;

      for (trainset_ndx = 0; trainset_ndx < _dataset.size (); trainset_ndx++)
      {
         const Exemplar<PatternSequence, PatternSequence> &exemplar = _dataset.at (
            trainset_ndx);

         PatternSequence ipattseq = exemplar.input ();
         PatternSequence tgtpattseq = exemplar.target_output ();

         double patt_err = 0;
         double prev_patt_err = 0;

         double patt_sse = 0;
         double seq_sse = 0;
         for (unsigned int pattern_ndx = 0; pattern_ndx < ipattseq.size ();
              pattern_ndx++)
         {
            const Pattern &ipatt = ipattseq.at (pattern_ndx);
            const Pattern &tgtpatt = tgtpattseq.at (pattern_ndx);

            // Present input pattern and get output
            const Pattern &opatt = neural_network (ipatt);

            if (pattern_ndx > 0)
            {
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
                  for (unsigned int i = 0; i < prev_opatt ().size (); i++)
                     pred_vec[i] = prev_opatt ().at (i);

                  if (pattern_ndx < ipattseq.size () - 1)
                     patt_err = -((gamma * opatt ().at (0) + tgtpatt ().at (0))
                                  - pred_vec[0]);
                  else
                     patt_err = -(tgtpatt ().at (0) - pred_vec[0]);

                  patt_sse = 0.5 * (patt_err * patt_err);
               }

               sample_count++;
               seq_sse += patt_sse;
            }

            prev_opatt = opatt;
         }
         //seq_sse = (ipattseq.size() > 0) ? seq_sse / ipattseq.size() : 0.0;

         global_performance += seq_sse;
      }
      //global_performance /= _dataset.size();

      if (sample_count > 0)
         global_performance /= sample_count;

      return global_performance;
   }

/* Matrix inversion routine.
 Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */
   template<class EFunc, class LRPolicy>
   bool TDCTrainer2<EFunc,
                    LRPolicy>::InvertMatrix (const boost::numeric::ublas::matrix<double> &input, boost::numeric::ublas::matrix<
      double> &inverse)
   {
      typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix;

      // create a working copy of the input
      boost::numeric::ublas::matrix<double> A (input);

      // create a permutation matrix for the LU-factorization
      pmatrix pm (A.size1 ());

      // perform LU-factorization
      int res = boost::numeric::ublas::lu_factorize (A, pm);
      if (res != 0)
         return false;

      // create identity matrix of "inverse"
      inverse.assign (boost::numeric::ublas::identity_matrix<double> (A.size1 ()));

      // backsubstitute to get the inverse
      boost::numeric::ublas::lu_substitute (A, pm, inverse);

      return true;
   }

   template<class EFunc, class LRPolicy>
   double TDCTrainer2<EFunc, LRPolicy>::sim2 (
      const DataSet<Exemplar<PatternSequence, PatternSequence> > &_dataset)
   {
      long epoch_no;
      unsigned int trainset_ndx;
      double perf;
      unsigned int sample_count = 0;

      vector<double> gradient (neural_network.get_output_size ());
      vector<double> ugradient (neural_network.get_output_size (), 1.0);
      Pattern prev_opatt = vector<double> (neural_network.get_output_size ());
      Pattern prev_ipatt;

      vector<double> cum_tgt_vec (neural_network.get_output_size ());
      vector<double> pred_vec (neural_network.get_output_size ());
      Pattern cum_tgtpatt;

      unsigned int insize = 0;
      vector<vector<double> > inerr = neural_network.get_input_error ();
      for (unsigned int i = 0; i < inerr.size (); i++)
         insize += inerr.at (i).size ();

      double global_performance = 0;

      const Exemplar<PatternSequence, PatternSequence> &temp_exemplar =
         _dataset.at (0);

      PatternSequence temp_ipattseq = temp_exemplar.input ();

      const Pattern &temp_ipatt = temp_ipattseq.at (0);
      unsigned int ipatt_size = temp_ipatt ().size ();

      vector<double> E_td_phi (ipatt_size, 0.0);
      vector<double> w (ipatt_size, 0.0);
      boost::numeric::ublas::matrix<double> E_cov_phi (ipatt_size, ipatt_size);
      boost::numeric::ublas::matrix<double> Inv_E_cov_phi (ipatt_size, ipatt_size);

      for (unsigned int i = 0; i < ipatt_size; i++)
         for (unsigned int j = 0; j < ipatt_size; j++)
            E_cov_phi (i, j) = 0.0;

      for (trainset_ndx = 0; trainset_ndx < _dataset.size (); trainset_ndx++)
      {
         const Exemplar<PatternSequence, PatternSequence> &exemplar = _dataset.at (
            trainset_ndx);

         PatternSequence ipattseq = exemplar.input ();
         PatternSequence tgtpattseq = exemplar.target_output ();

         double td_err = 0;
         double td_patt_err = 0;

         double patt_sse = 0;
         double seq_sse = 0;
         for (unsigned int pattern_ndx = 0; pattern_ndx < ipattseq.size ();
              pattern_ndx++)
         {
            const Pattern &ipatt = ipattseq.at (pattern_ndx);
            const Pattern &tgtpatt = tgtpattseq.at (pattern_ndx);

            vector<double> dEdB;
            Array<double> dEdW (1, insize);
            vector<double> prev_dEdB;
            Array<double> prev_dEdW (1, insize);

            prev_dEdB = dEdB;
            prev_dEdW = dEdW;

            // Present input pattern and get output
            const Pattern &opatt = neural_network (ipatt);

            // Save network gradients before accumulating new gradient
            neural_network.clear_error ();
            neural_network.backprop (ugradient);

            const vector<BaseLayer *> network_layers =
               neural_network.get_network_layers ();
            for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
            {
               BaseLayer &layer = *network_layers[ndx];
               const string &name = layer.name ();

               dEdB = layer.get_dEdB ();
               dEdW = layer.get_dEdW ();
            }

            if (pattern_ndx > 0)
            {
               if (pattern_ndx < ipattseq.size () - 1)
                  td_err = tgtpatt ().at (0) + gamma * opatt ().at (0)
                           - prev_opatt ().at (0);
               else
                  td_err = tgtpatt ().at (0) - prev_opatt ().at (0);

               for (unsigned in_ndx = 0; in_ndx < prev_ipatt ().size (); in_ndx++)
                  E_td_phi.at (in_ndx) += td_err * prev_dEdW.at (0, in_ndx);

               for (unsigned int i = 0; i < ipatt_size; i++)
                  for (unsigned int j = 0; j < ipatt_size; j++)
                     E_cov_phi (i, j) += prev_dEdW.at (0, i) * prev_dEdW.at (0, j);

               sample_count++;
            }

            prev_opatt = opatt;
            prev_ipatt = ipatt;
         }
      }

      if (sample_count > 0)
      {
         for (unsigned in_ndx = 0; in_ndx < ipatt_size; in_ndx++)
            E_td_phi.at (in_ndx) /= sample_count;

         for (unsigned int i = 0; i < ipatt_size; i++)
            for (unsigned int j = 0; j < ipatt_size; j++)
               E_cov_phi (i, j) /= sample_count;

         InvertMatrix (E_cov_phi, Inv_E_cov_phi);

         for (unsigned int i = 0; i < ipatt_size; i++)
            for (unsigned int j = 0; j < ipatt_size; j++)
               w.at (i) += Inv_E_cov_phi (i, j) * E_td_phi.at (j);

         for (unsigned int i = 0; i < ipatt_size; i++)
            global_performance += E_td_phi.at (i) * w.at (i);

         global_performance = sqrt (global_performance);
      }

      return global_performance;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_TDCTRAINER_H_ */
