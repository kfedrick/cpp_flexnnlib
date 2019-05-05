/*
 * GreedyActorCriticTrainer.h
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_GREEDYACTORCRITIC_TRAINER_H_
#define FLEX_NEURALNET_GREEDYACTORCRITIC_TRAINER_H_

#include <vector>
#include <string>

#include "GreedyActorCriticNet.h"
#include "EnvironmentSimulator.h"
#include "TrainingRecord.h"
#include "NetworkWeightsData.h"
#include "DeltaBarDeltaLearningRate.h"
#include "ConstantLearningRate.h"

using namespace std;

namespace flex_neuralnet
{

class GreedyActorCriticTrainer
{
public:
   GreedyActorCriticTrainer(GreedyActorCriticNet& _model);
   virtual ~GreedyActorCriticTrainer();

   /* ***************************************************
    *    Public setter methods
    */
   void set_max_epochs(long _epochs);
   void set_lambda(double l);
   void set_global_learning_rate(double _rate);
   void set_actor_learning_rate(double _rate);
   void set_critic_learning_rate(double _rate);
   void set_learning_momentum(double _rate);
   void set_performance_goal(double _perf);
   void set_min_gradient(double _min);
   void set_max_validation_failures(int _count);

   void set_predict_final_cost();
   void set_predict_cumulative_cost();

   void set_batch_mode();
   void set_minibatch_mode(unsigned int _size);
   void set_online_mode();

   void set_random_order_mode(bool _val);
   void set_report_frequency(int _freq);
   void set_verbose(bool _mode);

   void set_critic_batch_size(unsigned int _sz);
   void set_actor_batch_size(unsigned int _sz);

   /* ****************************************************
    *    Public getter methods
    */

   long max_epochs();
   double performance_goal();
   double min_gradient();
   int max_validation_failures();

   bool is_batch_mode();
   bool is_online_mode();
   bool is_minibatch_mode();

   bool is_random_order_mode();
   bool is_verbose_mode();
   unsigned int minibatch_size();
   int report_frequency();

   double global_perf();

   const TrainingRecord& training_record();

   /*
    * Activate the model for the specified environment
    */
   virtual double sim(EnvironmentSimulator* _env, unsigned int _sampleCount);
   virtual double sim2(EnvironmentSimulator* _env, unsigned int _sampleCount);

   virtual void train(EnvironmentSimulator* _trainingEnvSet, double _objVal);
   virtual double train_critic(EnvironmentSimulator* _trainingEnvSet, unsigned int _batchSize);
   virtual double train_actor(EnvironmentSimulator* _trainingEnvSet, double _objVal, unsigned int _batchSize);


private:
   void init_train();
   void init_training_epoch();
   virtual double train_exemplar(EnvironmentSimulator* _trainingEnv, double _objVal, bool _updateActorFlag);

   virtual void calc_network_adj(const BaseNeuralNet& _net, const vector<double>& errorv);

   void calc_layer_bias_adj(const BaseNeuralNet& _net, const BaseLayer& layer, unsigned int timeStep,
         vector<double>& biasDelta, const vector<double>& errorv);
   void calc_layer_weight_adj(const BaseNeuralNet& _net, const BaseLayer& layer, unsigned int timeStep,
         Array<double>& weightDelta, const vector<double>& errorv);

   virtual void activate_actor_critic(const Pattern& _currStateVec, const Pattern& _actionVec);

   void save_network_eligibility_trace();
   void save_network_eligibility_trace(const BaseNeuralNet& _net);

   void update_network_eligibility_trace();
   void update_network_eligibility_trace(const BaseNeuralNet& _net);

   void zero_network_eligibility_trace();
   void zero_network_eligibility_trace(const BaseNeuralNet& _net);

protected:

   /* ****************************************************
    *    Protected methods to save and restore network
    *    network weights
    */
   void save_weights(const string& _id);
   void save_weights(const string& _id, const BaseNeuralNet& _net);

   void restore_weights(const string& _id);
   void restore_weights(const string& _id, const BaseNeuralNet& _net);


   /* *****************************************************
    *    Protected helper methods for managing network
    *    weight training deltas and other weights cache
    */
   bool initialized;
   NetworkWeightsData& get_cached_network_weights(const string& _id, const BaseNeuralNet& _net);

   void alloc_network_weights_cache_entry(
         const string& _id);
   void alloc_network_weights_cache_entry(
         const string& _id, const BaseNeuralNet& _net);

   void apply_delta_network_weights();
   void apply_delta_network_weights(BaseNeuralNet& _net);

   void zero_delta_network_weights();
   void zero_delta_network_weights(const BaseNeuralNet& _net);

   void alloc_network_learning_rates();
   void apply_learning_rate_adjustments();
   void reduce_learning_rates(double val);

   /* *******************************************************
    *    Protected methods to manage performance trace
    */
   void clear_training_record();
   void update_training_record(unsigned int _epoch, unsigned int _stopSig, const TrainingRecord::Entry& _trEntry);


   /* *******************************************************
    *    Protected utility functions
    */
   int urand(int n);
   const string& genkey(const BaseNeuralNet& _net, const string& _id);

protected:

   /* ***********************************************
    *    static const data member
    */
   static const long default_max_epochs;
   static const double default_learning_momentum;
   static const double default_performance_goal;
   static const double default_min_gradient;
   static const int default_max_valid_fail;
   static const bool default_batch_mode;
   static const unsigned int default_batch_size;
   static const int default_report_frequency;

   static const string delta_network_weights_id;
   static const string previous_delta_network_weights_id;
   static const string adjusted_network_weights_id;
   static const string best_weights_id;
   static const string failback_weights_id;

   /* ***********************************************
    *    Stopping criteria parameters
    */
   long max_training_epochs;
   double perf_goal;
   double min_perf_gradient;
   int max_validation_fail;

   /* ************************************************
    *    Network training mode parameters
    */
   double learning_momentum;
   bool batch_mode;
   unsigned int batch_size;
   bool random_order_mode;
   bool verbose_mode;

   unsigned int critic_batch_size;
   unsigned int actor_batch_size;

   /* ************************************************
    *    Performance data reporting parameters
    */
   int report_freq;

private:

   enum Prediction_Mode
   {
      FINAL_COST, CUMULATIVE_COST
   };

   /* ***********************************************
    *    Error function
    */
   SumSquaredError error_func;

   /* ***********************************************
    *    The actor-critic model
    */
   GreedyActorCriticNet& actor_critic_model;

   /* ***********************************************
    *    Protected working storage data members
    */
   map<string, NetworkWeightsData> network_weights_cache;

   /* ************************************************
    *    Training performance record
    */
   TrainingRecord last_training_record;

   /* ************************************************
    *   learning rate policy
    */
     map<string, DeltaBarDeltaLearningRate*> network_learning_rates_map;
     //map<string, ConstantLearningRate*> network_learning_rates_map;

     double lambda;
     Prediction_Mode predict_mode;

     map<string, Array<double> > etrace_dAdB_map;
     map<string, Array<double> > etrace_dAdW_map;

     map<string, Array<double> > cumulative_dAdN_map;
     map<string, Array<double> > cumulative_dNdW_map;

     map<string, Array<double> > prev_etrace_dAdB_map;
     map<string, Array<double> > prev_etrace_dAdW_map;

     map<string, Array<double> > prev_cumulative_dAdN_map;
     map<string, Array<double> > prev_cumulative_dNdW_map;
};

inline
void GreedyActorCriticTrainer::set_max_epochs(long _epochs)
{
   max_training_epochs = _epochs;
}

inline
void GreedyActorCriticTrainer::set_lambda(double l)
{
   lambda = l;
}

inline
void GreedyActorCriticTrainer::set_learning_momentum(double _rate)
{
   if (_rate < 0)
   {
      ostringstream err_str;
      err_str << "BaseTrainer::set_learning_momentum() Error : " << endl;
      err_str << "   Attempt to set learning momentum to illegal value - " << _rate << "." << endl;
      err_str << "   The value must be greater than or equal to zero." << endl;
      throw invalid_argument(err_str.str());
   }
   learning_momentum = _rate;
}

inline
void GreedyActorCriticTrainer::set_performance_goal(double _perf)
{
   perf_goal = _perf;
}

inline
void GreedyActorCriticTrainer::set_min_gradient(double _min)
{
   min_perf_gradient = _min;
}

inline
void GreedyActorCriticTrainer::set_max_validation_failures(int _count)
{
   max_validation_fail = _count;
}

inline
void GreedyActorCriticTrainer::set_batch_mode()
{
   batch_mode = true;
   batch_size = 0;
}

inline
void GreedyActorCriticTrainer::set_minibatch_mode(unsigned int _size)
{
   batch_mode = true;
   batch_size = _size;
}

inline
void GreedyActorCriticTrainer::set_online_mode()
{
   batch_mode = false;
}

inline
void GreedyActorCriticTrainer::set_random_order_mode(bool _val)
{
   random_order_mode = _val;
}

inline
void GreedyActorCriticTrainer::set_verbose(bool _mode)
{
   verbose_mode = _mode;
}

inline
void GreedyActorCriticTrainer::set_report_frequency(int _freq)
{
   report_freq = _freq;
}

inline
void GreedyActorCriticTrainer::set_critic_batch_size(unsigned int _sz)
{
   critic_batch_size = _sz;
}

inline
void GreedyActorCriticTrainer::set_actor_batch_size(unsigned int _sz)
{
   actor_batch_size = _sz;
}

inline
long GreedyActorCriticTrainer::max_epochs()
{
   return max_training_epochs;
}

inline
double GreedyActorCriticTrainer::performance_goal()
{
   return perf_goal;
}

inline
double GreedyActorCriticTrainer::min_gradient()
{
   return min_perf_gradient;
}

inline
int GreedyActorCriticTrainer::max_validation_failures()
{
   return max_validation_fail;
}

inline
bool GreedyActorCriticTrainer::is_batch_mode()
{
   return batch_mode && (batch_size == 0);
}

inline
bool GreedyActorCriticTrainer::is_minibatch_mode()
{
   return batch_mode && (batch_size > 0);
}

inline
bool GreedyActorCriticTrainer::is_online_mode()
{
   return !batch_mode;
}

inline
unsigned int GreedyActorCriticTrainer::minibatch_size()
{
   return batch_size;
}

inline
bool GreedyActorCriticTrainer::is_random_order_mode()
{
   return random_order_mode;
}

inline
bool GreedyActorCriticTrainer::is_verbose_mode()
{
   return verbose_mode;
}

inline
int GreedyActorCriticTrainer::report_frequency()
{
   return report_freq;
}

inline
double GreedyActorCriticTrainer::global_perf()
{
   return last_training_record.best_training_perf();
}



inline
const TrainingRecord& GreedyActorCriticTrainer::training_record()
{
   return last_training_record;
}

inline
void GreedyActorCriticTrainer::clear_training_record()
{
   last_training_record.clear();
}

inline
const string& GreedyActorCriticTrainer::genkey(const BaseNeuralNet& _net, const string& _id)
{
   static string key;

   key = _net.name() + "_" + _id;
   return key;
}

inline
void GreedyActorCriticTrainer::activate_actor_critic(const Pattern& _currStateVec, const Pattern& _actionVec)
{

}


} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_GREEDYACTORCRITIC_TRAINER_H_ */