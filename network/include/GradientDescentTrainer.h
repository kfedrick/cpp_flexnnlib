/*
 * GradientDescentTrainer.h
 *
 *  Created on: Feb 26, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_GDTRAINER_H_
#define FLEX_NEURALNET_GDTRAINER_H_

#include "BaseTrainer.h"

#include <iostream>
#include <vector>
#include <cfloat>

using namespace std;

namespace flex_neuralnet
{

template<class EFunc, class LRPolicy>
class GradientDescentTrainer: public BaseTrainer
{
public:
   GradientDescentTrainer(BaseNeuralNet& _net);
   virtual ~GradientDescentTrainer();

   void alloc_network_learning_rates();

   void init_train(const DataSet<Exemplar<Pattern, Pattern> >& trnset,
         const DataSet<Exemplar<Pattern, Pattern> >& vldset = DataSet<
               Exemplar<Pattern, Pattern> >(),
         const DataSet<Exemplar<Pattern, Pattern> >& tstset = DataSet<
               Exemplar<Pattern, Pattern> >());
   void train(const DataSet<Exemplar<Pattern, Pattern> >& trnset,
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

   void init_training_epoch();
   double train_exemplar(const Exemplar<Pattern, Pattern>& exemplar);
   LRPolicy& get_learning_rates();

protected:
   EFunc& get_errorfunc();

private:
   void calc_layer_bias_adj(const BaseLayer& layer, unsigned int timeStep,
         vector<double>& biasDelta);
   void calc_layer_weight_adj(const BaseLayer& layer, unsigned int timeStep,
         Array<double>& weightDelta);

private:

   /* ***********************************************
    *    Error function
    */
   EFunc error_func;
};

template<class EFunc, class LRPolicy>
GradientDescentTrainer<EFunc, LRPolicy>::GradientDescentTrainer(
      BaseNeuralNet& _net) :
      BaseTrainer(_net)
{
   set_verbose(false);

   alloc_network_learning_rates();
   set_global_learning_rate(0.01);
}

template<class EFunc, class LRPolicy>
GradientDescentTrainer<EFunc, LRPolicy>::~GradientDescentTrainer()
{
   // TODO Auto-generated destructor stub
}

template<class EFunc, class LRPolicy>
void GradientDescentTrainer<EFunc, LRPolicy>::alloc_network_learning_rates()
{
   if (network_learning_rates == NULL)
      network_learning_rates = new LRPolicy(neural_network);
}

template<class EFunc, class LRPolicy>
EFunc& GradientDescentTrainer<EFunc, LRPolicy>::get_errorfunc()
{
   return error_func;
}

template<class EFunc, class LRPolicy>
LRPolicy& GradientDescentTrainer<EFunc, LRPolicy>::get_learning_rates()
{
   LRPolicy* learning_rates = dynamic_cast<LRPolicy*>(network_learning_rates);
   return *learning_rates;
}

template<class EFunc, class LRPolicy>
void GradientDescentTrainer<EFunc, LRPolicy>::train(
      const DataSet<Exemplar<Pattern, Pattern> >& trnset,
      const DataSet<Exemplar<Pattern, Pattern> >& vldset,
      const DataSet<Exemplar<Pattern, Pattern> >& tstset)
{
   int stop_status = TRAINING;
   long epoch_no;
   unsigned int trainset_ndx;
   unsigned int mini_batch_count;

   double perf;
   double global_performance = 0;
   double prev_valid_perf = DBL_MAX;
   int validation_failures = 0;
   unsigned int consecutive_failback_count = 0;
   bool new_best_flag = false;

   unsigned int permute_ndx;
   vector<unsigned int> presentation_order(trnset.size());

   init_train(trnset, vldset, tstset);
   initialize_presentation_order(presentation_order);

   for (epoch_no = 0; stop_status == TRAINING; epoch_no++)
   {
      global_performance = 0;
      init_training_epoch();

      if (is_random_order_mode())
         permute_presentation_order(presentation_order);

      mini_batch_count = 0;
      for (trainset_ndx = 0; trainset_ndx < trnset.size(); trainset_ndx++)
      {
         permute_ndx = presentation_order[trainset_ndx];

         // cout << "trn index " << trainset_ndx << "; perm index " << permute_ndx << endl;

         perf = train_exemplar(trnset.at(permute_ndx));
         global_performance += perf;

         mini_batch_count++;

         if (is_online_mode())
         {
            apply_delta_network_weights();
            zero_delta_network_weights();
         }
         else if (is_minibatch_mode() && mini_batch_count > minibatch_size())
         {
            apply_delta_network_weights();
            zero_delta_network_weights();
            mini_batch_count = 0;
         }
      }
      global_performance /= trnset.size();

      network_learning_rates->apply_learning_rate_adjustments();

      if (is_batch_mode() || (is_minibatch_mode() && mini_batch_count > 0))
      {
         apply_delta_network_weights();
      }

      if (is_verbose_mode())
         cout << "global perf(" << epoch_no << ") = " << global_performance
               << " " << trnset.size() << endl;

      // TODO - calculate performance gradient
      // TODO - if we did weight adjustment with rollback then I'd need to do it here

      TrainingRecord::Entry& perf_rec = calc_performance_data(epoch_no + 1,
            trnset, vldset, tstset);

      // TODO - make the threshold a variable that can be set
      // TODO - make the learning rate reduction a variable that can be set
      // if error increases by 3% then back out weight changes
      if ( (perf_rec.training_perf() - global_performance)/ global_performance > 0.03)
      {
         cout << "fail back weights because of large increase in error." << endl;

         consecutive_failback_count++;
         if (consecutive_failback_count > 10)
         {
            cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
            update_training_record(epoch_no + 1, MAX_CONSECUTIVE_FAILBACK_STOP, perf_rec);
            break;
         }

         restore_weights(failback_weights_id);
         network_learning_rates->reduce_learning_rate(0.4);
         epoch_no--;

         continue;
      }

      // Clear consecutive failback count
      consecutive_failback_count = 0;

      if (is_best_perf(perf_rec))
         save_weights(best_weights_id);

      stop_status = is_training_complete(perf_rec, prev_valid_perf,
            validation_failures);

      update_training_record(epoch_no + 1, stop_status, perf_rec);
   }

   // Restore weights to the best training epoch
   restore_weights(best_weights_id);
}

template<class EFunc, class LRPolicy>
void GradientDescentTrainer<EFunc, LRPolicy>::train(
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _trnset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _vldset,
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& _tstset)
{

}

template<class EFunc, class LRPolicy>
double GradientDescentTrainer<EFunc, LRPolicy>::train_exemplar(
      const Exemplar<Pattern, Pattern>& exemplar)
{
   const Pattern& inpattern = exemplar.input();
   const Pattern& tgtvec = exemplar.target_output();

   // Present input pattern and get output
   const Pattern& netout = neural_network(exemplar.input());

   // Calculate the output error
   double isse;
   vector<double> gradient(netout().size());

   error_func(isse, gradient, netout, tgtvec);

   neural_network.clear_error();

   // cout << "isse " << isse << endl;

   // Backprop the error through the network
   /*
   for (unsigned int i = 0; i < gradient.size(); i++)
      gradient[i] = -gradient[i];
      */

   neural_network.backprop(gradient);

   network_learning_rates->update_learning_rate_adjustments(1);

   NetworkWeightsData& network_deltas = get_cached_network_weights(
         delta_network_weights_id);

   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      LayerWeightsData& layer_deltas = network_deltas.layer_weights(name);

      calc_layer_bias_adj(layer, 1, layer_deltas.biases);
      calc_layer_weight_adj(layer, 1, layer_deltas.weights);
   }

   return isse;
}

template<class EFunc, class LRPolicy>
void GradientDescentTrainer<EFunc, LRPolicy>::calc_layer_bias_adj(
      const BaseLayer& layer, unsigned int timeStep, vector<double>& biasDelta)
{
   const vector<double>& errorv = layer.get_error(timeStep);
   const Array<double>& dAdB = layer.get_dAdB(timeStep);
   const vector<double>& dEdB = layer.get_dEdB(timeStep);

   // Get the learning rates for the biases
   vector<double> layer_bias_lr = network_learning_rates->get_bias_learning_rates().at(layer.name());
   for (unsigned int netin_ndx = 0; netin_ndx < dAdB.colDim(); netin_ndx++)
   {
      biasDelta.at(netin_ndx) += -layer_bias_lr[netin_ndx] * dEdB.at(netin_ndx);
      /*
      //biasDelta.at(in_ndx) = 0;
      for (unsigned int out_ndx = 0; out_ndx < dAdB.rowDim(); out_ndx++)
         biasDelta.at(netin_ndx) += -layer_bias_lr[out_ndx] * errorv.at(out_ndx)
               * dAdB.at(out_ndx, netin_ndx);
       */
   }
}

template<class EFunc, class LRPolicy>
void GradientDescentTrainer<EFunc, LRPolicy>::calc_layer_weight_adj(
      const BaseLayer& layer, unsigned int timeStep, Array<double>& weightDelta)
{
   const vector<double>& errorv = layer.get_error(timeStep);
   const Array<double>& dNdW = layer.get_dNdW(timeStep);
   const Array<double>& dAdN = layer.get_dAdN(timeStep);
   const Array<double>& dEdW = layer.get_dEdW(timeStep);

   // Get the learning rates for the weights
   Array<double> layer_weights_lr = network_learning_rates->get_weight_learning_rates().at(layer.name());

   unsigned int layer_input_size = layer.input_size();
   for (unsigned int out_ndx = 0; out_ndx < layer.size(); out_ndx++)
   {
      for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
      {
         weightDelta.at(out_ndx, in_ndx) +=
                           -layer_weights_lr[out_ndx][in_ndx] * dEdW.at(out_ndx, in_ndx);
         /*
         //weightDelta.at(out_ndx, in_ndx) = 0;
         for (unsigned int netin_ndx = 0; netin_ndx < layer.size(); netin_ndx++)
         {
            weightDelta.at(out_ndx, in_ndx) +=
                  -layer_weights_lr[netin_ndx][in_ndx] * errorv.at(out_ndx)
                        * dAdN.at(out_ndx, netin_ndx)
                        * dNdW.at(netin_ndx, in_ndx);
            //cout << weightDelta.at(out_sz, in_sz) << " " << endl;
         }
         */
      }
      //cout << endl;
   }
   //cout << endl << "^^^^ weight delta ^^^^" << endl;
}

/*
 * Instantiate and initialize any data structures needed to train the network as required.
 * For example structures to hold and accumulate the weight and bias deltas.
 */
template<class EFunc, class LRPolicy>
void GradientDescentTrainer<EFunc, LRPolicy>::init_train(
      const DataSet<Exemplar<Pattern, Pattern> >& _trnset,
      const DataSet<Exemplar<Pattern, Pattern> >& _vldset,
      const DataSet<Exemplar<Pattern, Pattern> >& _tstset)
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

   save_weights(best_weights_id);
   save_weights(failback_weights_id);
}

/*
 * Perform any initialization required for the new training epoch. For example clear
 * all data structures required to accumulate the new global network error, the weight
 * and bias deltas and etc.
 */
template<class EFunc, class LRPolicy>
void GradientDescentTrainer<EFunc, LRPolicy>::init_training_epoch()
{
   neural_network.clear_error();
   zero_delta_network_weights();
   save_weights(failback_weights_id);
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_GDTRAINER_H_ */
