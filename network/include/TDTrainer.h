/*
 * TDTrainer.h
 *
 *  Created on: Feb 26, 2014
 *      Author: kfedrick
 *
 *  Temporal Difference trainer for original Sutton
 *  formula of both final cost and cost-to-go.
 *
 *  NOTE: The global objective function does not quite
 *  reflect the value being minimized so it can't be
 *  effectively used for early stopping. In the early
 *  stages of learning it decreases but after a while
 *  it starts drifting.
 */

#ifndef FLEX_NEURALNET_TDTRAINER_H_
#define FLEX_NEURALNET_TDTRAINER_H_

#include "BaseTrainer.h"
#include <cfloat>

namespace flex_neuralnet
{

template<class EFunc, class LRPolicy>
class TDTrainer: public BaseTrainer
{
public:
   TDTrainer(BaseNeuralNet& _net);
   virtual ~TDTrainer();

   void alloc_network_learning_rates();

   LRPolicy& get_learning_rates();

   void set_predict_final_cost();
   void set_predict_cumulative_cost();

   void set_lambda(double l);
   void set_gamma(double l);

   virtual void train(const DataSet<Exemplar<Pattern, Pattern> >& trnset,
         const DataSet<Exemplar<Pattern, Pattern> >& vldset = DataSet<
               Exemplar<Pattern, Pattern> >(),
         const DataSet<Exemplar<Pattern, Pattern> >& tstset = DataSet<
               Exemplar<Pattern, Pattern> >());

   void train(
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _trnset,
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _vldset =
               DataSet<Exemplar<PatternSequence, PatternSequence> >(),
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _tstset =
               DataSet<Exemplar<PatternSequence, PatternSequence> >());

   double sim(
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _dataset);

   TrainingRecord::Entry& calc_performance_data(unsigned int _epoch,
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _trnset,
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _vldset =
               DataSet<Exemplar<PatternSequence, PatternSequence> >(),
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _tstset =
               DataSet<Exemplar<PatternSequence, PatternSequence> >());

protected:

   void init_train(
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _trnset,
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _vldset =
               DataSet<Exemplar<PatternSequence, PatternSequence> >(),
         const DataSet<Exemplar<PatternSequence, PatternSequence> >& _tstset =
               DataSet<Exemplar<PatternSequence, PatternSequence> >());

   void init_training_epoch();

   double train_exemplar(
         const Exemplar<PatternSequence, PatternSequence>& _exemplar);

   BaseNeuralNet& get_neuralnet();
   EFunc& get_errorfunc();

private:
   void save_eligibility_trace();
   void update_eligibility_trace();
   void zero_eligibility_trace();

   void calc_layer_bias_adj(const BaseLayer& layer, unsigned int timeStep,
         vector<double>& biasDelta, const vector<double>& errorv);
   void calc_layer_weight_adj(const BaseLayer& layer, unsigned int timeStep,
         Array<double>& weightDelta, const vector<double>& errorv);

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
   BaseNeuralNet& neural_network;

   /* ***********************************************
    *    Learning rate parameters
    */
   double lambda;
   double gamma;

   Prediction_Mode predict_mode;

   map<string, Array<double> > etrace_dAdB_map;
   map<string, Array<double> > etrace_dAdW_map;

   map<string, Array<double> > prev_etrace_dAdB_map;
   map<string, Array<double> > prev_etrace_dAdW_map;

   unsigned int pattern_no;
};

template<class EFunc, class LRPolicy>
TDTrainer<EFunc, LRPolicy>::TDTrainer(BaseNeuralNet& _net) :
      neural_network(_net), BaseTrainer(_net)
{
   set_batch_mode();
   set_predict_final_cost();
   set_random_order_mode(true);
   set_verbose(false);
   set_global_learning_rate(0.05);
   set_lambda(0.5);
}

template<class EFunc, class LRPolicy>
TDTrainer<EFunc, LRPolicy>::~TDTrainer()
{
   // TODO Auto-generated destructor stub
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::alloc_network_learning_rates()
{
   if (network_learning_rates == NULL)
      network_learning_rates = new LRPolicy(neural_network);
}

template<class EFunc, class LRPolicy>
BaseNeuralNet& TDTrainer<EFunc, LRPolicy>::get_neuralnet()
{
   return neural_network;
}

template<class EFunc, class LRPolicy>
EFunc& TDTrainer<EFunc, LRPolicy>::get_errorfunc()
{
   return error_func;
}

template<class EFunc, class LRPolicy>
LRPolicy& TDTrainer<EFunc, LRPolicy>::get_learning_rates()
{
   LRPolicy* learning_rates = dynamic_cast<LRPolicy*>(network_learning_rates);
   return *learning_rates;
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::set_lambda(double l)
{
   lambda = l;
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::set_gamma(double g)
{
   gamma = g;
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::set_predict_final_cost()
{
   predict_mode = FINAL_COST;
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::set_predict_cumulative_cost()
{
   predict_mode = CUMULATIVE_COST;
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::train(
      const DataSet<Exemplar<Pattern, Pattern> >& trnset,
      const DataSet<Exemplar<Pattern, Pattern> >& vldset,
      const DataSet<Exemplar<Pattern, Pattern> >& tstset)
{

}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::train(
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _trnset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _vldset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _tstset)
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
   vector<unsigned int> presentation_order(_trnset.size());

   cout << "training set size " << _trnset.size() << endl;

   init_train(_trnset, _vldset, _tstset);
   initialize_presentation_order(presentation_order);

   for (epoch_no = 0; stop_status == TRAINING; epoch_no++)
   {
      global_performance = 0;
      init_training_epoch();

      if (is_random_order_mode())
         permute_presentation_order(presentation_order);

      int cnt = 0;
      for (trainset_ndx = 0; trainset_ndx < _trnset.size(); trainset_ndx++)
      {
         cnt++;
         permute_ndx = presentation_order[trainset_ndx];
         pattern_no = trainset_ndx; // ????

         perf = train_exemplar(_trnset.at(permute_ndx));
         global_performance += perf;

         if (is_online_mode())
         {
            apply_delta_network_weights();
            zero_delta_network_weights();
            zero_eligibility_trace();
         }
      }
      global_performance /= _trnset.size();

      network_learning_rates->apply_learning_rate_adjustments();

      if (is_batch_mode())
         apply_delta_network_weights();

      if (is_verbose_mode())
         cout << "global perf(" << epoch_no << ") = " << global_performance
               << endl;

      // TODO - calculate performance gradient
      // TODO - if we did weight adjustment with rollback then I'd need to do it here

      TrainingRecord::Entry& perf_rec = calc_performance_data(epoch_no + 1,
            _trnset, _vldset, _tstset);

      // TODO - make the threshold a variable that can be set
      // TODO - make the learning rate reduction a variable that can be set
      // if error increases by 3% then back out weight changes
      if ((perf_rec.training_perf() - prev_global_performance)
            / prev_global_performance > 0.05)
      {
         cout << "fail back weights because of large increase in error. "
               << prev_global_performance << " => " << perf_rec.training_perf()
               << endl;

         consecutive_failback_count++;
         if (consecutive_failback_count > 10)
         {
            cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
            update_training_record(epoch_no + 1, MAX_CONSECUTIVE_FAILBACK_STOP,
                  perf_rec);
            break;
         }

         restore_weights(failback_weights_id);
         network_learning_rates->reduce_learning_rate(0.4);
         epoch_no--;

         continue;
      }
      else
      {
         prev_global_performance = perf_rec.training_perf();
         consecutive_failback_count = 0;
      }

      if (is_best_perf(perf_rec))
         save_weights(best_weights_id);

      stop_status = is_training_complete(perf_rec, prev_valid_perf,
            validation_failures);
      update_training_record(epoch_no + 1, stop_status, perf_rec);
   }

   // Restore weights to the best training epoch
   restore_weights(best_weights_id);
   cout << "terminated training on status " << stop_status << endl;
}

template<class EFunc, class LRPolicy>
double TDTrainer<EFunc, LRPolicy>::train_exemplar(
      const Exemplar<PatternSequence, PatternSequence>& exemplar)
{
   vector<double> egradient(neural_network.get_output_size());
   vector<double> ugradient(neural_network.get_output_size(), 1.0);

   Pattern prev_opatt = vector<double>(neural_network.get_output_size());
   Pattern opatt = vector<double>(neural_network.get_output_size());

   PatternSequence ipattseq = exemplar.input();
   PatternSequence tgtpattseq = exemplar.target_output();

   NetworkWeightsData& network_deltas = get_cached_network_weights(
         delta_network_weights_id);

   neural_network.clear_error();
   zero_eligibility_trace();

   double patt_err = 0;
   double prev_patt_err = 0;

   double patt_sse = 0;
   double seq_sse = 0;
   unsigned int pattern_ndx;
   Pattern prev_ipatt;

   vector<double> cum_tgt_vec(neural_network.get_output_size());
   vector<double> pred_vec(neural_network.get_output_size());
   Pattern cum_tgtpatt;

   //cout << "********************" << endl;
   for (pattern_ndx = 0; pattern_ndx < ipattseq.size(); pattern_ndx++)
   {
      const Pattern& ipatt = ipattseq.at(pattern_ndx);
      const Pattern& tgtpatt = tgtpattseq.at(pattern_ndx);

      // Present input pattern and get output

      const Pattern& netout = neural_network(ipatt);
      opatt = netout;

      // Save network gradients before accumulating new gradient
      save_eligibility_trace();

      neural_network.clear_error();
      neural_network.backprop(ugradient);

      update_eligibility_trace();

      // The first pattern can't learn anything
      if (pattern_ndx > 0)
      {
         prev_opatt = neural_network(prev_ipatt);

         if (predict_mode == FINAL_COST)
         {
            if (tgtpatt().size() > 0)
               error_func(patt_sse, egradient, prev_opatt, tgtpatt);
            else
               error_func(patt_sse, egradient, prev_opatt, opatt);
         }
         else if (predict_mode == CUMULATIVE_COST)
         {
            for (unsigned int i = 0; i < prev_opatt().size(); i++)
               pred_vec[i] = prev_opatt().at(i);

            prev_patt_err = patt_err;

            if (pattern_ndx < ipattseq.size() - 1)
            {
               patt_err =
                  -((gamma * opatt().at(0) + tgtpatt().at(0)) - pred_vec[0]);
            }
            else
            {
               patt_err = -(tgtpatt().at(0) - pred_vec[0]);
            }

            egradient[0] = patt_err;
            patt_sse = 0.5 * (patt_err * patt_err);
         }

         seq_sse += patt_sse;

         network_learning_rates->update_learning_rate_adjustments(1);

         /*
          * Calculate the weight updates for the PREVIOUS network activation
          */
         const vector<BaseLayer*> network_layers =
               neural_network.get_network_layers();
         for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
         {
            BaseLayer& layer = *network_layers[ndx];
            const string& name = layer.name();

            LayerWeightsData& layer_deltas = network_deltas.layer_weights(name);

            calc_layer_bias_adj(layer, 1, layer_deltas.biases, egradient);
            calc_layer_weight_adj(layer, 1, layer_deltas.weights, egradient);
         }
      }

      // Save current input
      prev_ipatt = ipatt;

      /*
       for (unsigned int i = 0; i < prev_outputv.size(); i++)
       prev_outputv[i] = netout[i];
       */
   } /* loop through patterns */

   seq_sse = (ipattseq.size() > 0) ? seq_sse / ipattseq.size() : 0;

   return seq_sse;
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::save_eligibility_trace()
{
   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      // Save elibigility dAdB
      Array<double>& etrace_dAdB = etrace_dAdB_map[name];
      Array<double>& prev_etrace_dAdB = prev_etrace_dAdB_map[name];

      prev_etrace_dAdB = etrace_dAdB;

      // Save cumulative dAdW
      Array<double>& etrace_dAdW = etrace_dAdW_map[name];
      Array<double>& prev_etrace_dAdW = prev_etrace_dAdW_map[name];

      prev_etrace_dAdW = etrace_dAdW;
   }
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::update_eligibility_trace()
{
   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      double prev_sum;
      double temp = 0;

      Array<double>& etrace_dAdB = etrace_dAdB_map[name];
      const vector<double>& dEdB = layer.get_dEdB();

      for (unsigned int bias_ndx = 0; bias_ndx < dEdB.size(); bias_ndx++)
      {
         prev_sum = etrace_dAdB.at(bias_ndx, bias_ndx);
         etrace_dAdB.at(bias_ndx, bias_ndx) = dEdB.at(bias_ndx)
               + lambda * prev_sum;
      }

      Array<double>& etrace_dAdW = etrace_dAdW_map[name];
      const Array<double>& dEdW = layer.get_dEdW();

      unsigned int layer_input_size = layer.input_size();
      for (unsigned int out_ndx = 0; out_ndx < layer.size(); out_ndx++)
      {
         for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
         {
            prev_sum = etrace_dAdW.at(out_ndx, in_ndx);
            etrace_dAdW.at(out_ndx, in_ndx) = dEdW.at(out_ndx, in_ndx)
                  + lambda * prev_sum;
         }
      }
   }
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::calc_layer_bias_adj(
      const BaseLayer& layer, unsigned int timeStep, vector<double>& biasDelta,
      const vector<double>& errorv)
{
   Array<double>& etrace_dAdB = prev_etrace_dAdB_map[layer.name()];

   // Get the learning rates for the biases
   vector<double> layer_bias_lr =
         network_learning_rates->get_bias_learning_rates().at(layer.name());

   for (unsigned int bias_ndx = 0; bias_ndx < biasDelta.size(); bias_ndx++)
   {
      for (unsigned int err_ndx = 0; err_ndx < errorv.size(); err_ndx++)
         biasDelta.at(bias_ndx) += -layer_bias_lr[bias_ndx] * errorv.at(err_ndx)
               * etrace_dAdB.at(bias_ndx, bias_ndx);
   }
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::calc_layer_weight_adj(
      const BaseLayer& layer, unsigned int timeStep, Array<double>& weightDelta,
      const vector<double>& errorv)
{
   Array<double>& etrace_dAdW = prev_etrace_dAdW_map[layer.name()];

   // Get the learning rates for the weights
   Array<double> layer_weights_lr =
         network_learning_rates->get_weight_learning_rates().at(layer.name());

   unsigned int layer_size = layer.size();
   unsigned int layer_input_size = layer.input_size();

   for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
      for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
         for (unsigned int err_ndx = 0; err_ndx < errorv.size(); err_ndx++)
            weightDelta.at(out_ndx, in_ndx) += -layer_weights_lr.at(out_ndx,
                  in_ndx) * errorv.at(err_ndx)
                  * etrace_dAdW.at(out_ndx, in_ndx);
}

/*
 * Instantiate and initialize any data structures needed to train the network as required.
 * For example structures to hold and accumulate the weight and bias deltas.
 */
template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::init_train(
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _trnset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _vldset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _tstset)
{
   neural_network.clear_error();
   zero_delta_network_weights();

   network_learning_rates->reset();

   TrainingRecord::Entry& entry = calc_performance_data(0, _trnset, _vldset,
         _tstset);

   clear_training_record();
   unsigned int epoch = 0;
   unsigned int stop_sig = BaseTrainer::TRAINING;
   update_training_record(epoch, stop_sig, entry);

   save_weights(failback_weights_id);
   save_weights(best_weights_id);

   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      const string& name = layer.name();
      unsigned int output_sz = layer.size();
      unsigned int input_sz = layer.input_size();

      etrace_dAdB_map[name] = Array<double>();
      etrace_dAdB_map[name].resize(output_sz, output_sz);

      etrace_dAdW_map[name] = Array<double>();
      etrace_dAdW_map[name].resize(output_sz, input_sz);

      prev_etrace_dAdB_map[name] = Array<double>();
      prev_etrace_dAdB_map[name].resize(output_sz, output_sz);

      prev_etrace_dAdW_map[name] = Array<double>();
      prev_etrace_dAdW_map[name].resize(output_sz, input_sz);
   }

   //   prev_outputv.resize(neural_network.get_output_size());
}

/*
 * Perform any initialization required for the new training epoch. For example clear
 * all data structures required to accumulate the new global network error, the weight
 * and bias deltas and etc.
 */
template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::init_training_epoch()
{
   neural_network.clear_error();
   zero_delta_network_weights();
   zero_eligibility_trace();
   save_weights(failback_weights_id);
}

template<class EFunc, class LRPolicy>
void TDTrainer<EFunc, LRPolicy>::zero_eligibility_trace()
{
   /* ******************************************
    *    Zero bias and weight deltas
    */
   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      const string& name = layer.name();

      Array<double>& etrace_dAdB = etrace_dAdB_map[name];
      Array<double>& etrace_dAdW = etrace_dAdW_map[name];

      etrace_dAdB = 0;
      etrace_dAdW = 0;
   }
}

template<class EFunc, class LRPolicy>
TrainingRecord::Entry& TDTrainer<EFunc, LRPolicy>::calc_performance_data(
      unsigned int _epoch,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _trnset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _vldset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _tstset)
{
   static TrainingRecord::Entry entry(_epoch);
   double perf;

   entry.set_epoch(_epoch);

   perf = sim(_trnset);
   entry.set_training_perf(perf);

   if (_vldset.size() > 0)
   {
      perf = sim(_vldset);
      entry.set_validation_perf(perf);
   }

   if (_tstset.size() > 0)
   {
      perf = sim(_tstset);
      entry.set_test_perf(perf);
   }

   return entry;
}

template<class EFunc, class LRPolicy>
double TDTrainer<EFunc, LRPolicy>::sim(
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _dataset)
{
   long epoch_no;
   unsigned int trainset_ndx;
   double perf;
   unsigned int sample_count = 0;

   vector<double> gradient(neural_network.get_output_size());
   Pattern prev_opatt = vector<double>(neural_network.get_output_size());

   vector<double> cum_tgt_vec(neural_network.get_output_size());
   vector<double> pred_vec(neural_network.get_output_size());
   Pattern cum_tgtpatt;

   double global_performance = 0;

   for (trainset_ndx = 0; trainset_ndx < _dataset.size(); trainset_ndx++)
   {
      const Exemplar<PatternSequence, PatternSequence>& exemplar = _dataset.at(
            trainset_ndx);

      PatternSequence ipattseq = exemplar.input();
      PatternSequence tgtpattseq = exemplar.target_output();

      double patt_err = 0;
      double prev_patt_err = 0;

      double patt_sse = 0;
      double seq_sse = 0;
      for (unsigned int pattern_ndx = 0; pattern_ndx < ipattseq.size();
            pattern_ndx++)
      {
         const Pattern& ipatt = ipattseq.at(pattern_ndx);
         const Pattern& tgtpatt = tgtpattseq.at(pattern_ndx);

         // Present input pattern and get output
         const Pattern& opatt = neural_network(ipatt);

         if (pattern_ndx > 0)
         {
            // Calculate the output error
            if (predict_mode == FINAL_COST)
            {
               if (tgtpatt().size() > 0)
                  error_func(patt_sse, gradient, prev_opatt, tgtpatt);
               else
                  error_func(patt_sse, gradient, prev_opatt, opatt);
            }
            else if (predict_mode == CUMULATIVE_COST)
            {
               // Copy pattern output to vector
               for (unsigned int i = 0; i < prev_opatt().size(); i++)
                  pred_vec[i] = prev_opatt().at(i);

               prev_patt_err = patt_err;

               /*
                * This presumes a single instance of the terminal state at
                * the end of the sequence. It MAY work with multiple instances
                * of the terminal state with a target transition cost of zero.
                */
               if (pattern_ndx < ipattseq.size() - 1)
                  patt_err =
                     -((gamma * opatt().at(0) + tgtpatt().at(0)) - pred_vec[0]);
               else
                  patt_err = -(tgtpatt().at(0) - pred_vec[0]);

               patt_sse = 0.5 * (patt_err * patt_err);
            }

            sample_count++;
            seq_sse += patt_sse;
         }

         prev_opatt = opatt;
      }
      seq_sse = (ipattseq.size() > 0) ? seq_sse / ipattseq.size() : 0.0;

      global_performance += seq_sse;
   }
   global_performance /= _dataset.size();

   //if (sample_count > 0)
   //   global_performance /= sample_count;

   return global_performance;
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_TDTRAINER_H_ */
