/*
 * BaseTrainer.h
 *
 *  Created on: Mar 27, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_BASETRAINER_H_
#define FLEX_NEURALNET_BASETRAINER_H_

#include "Array.h"
#include "BaseNeuralNet.h"
#include "LearningRatePolicy.h"
#include "TrainingRecord.h"
#include "DataSet.h"
#include "OutputErrorFunctor.h"
#include "NetworkWeightsData.h"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <set>

using namespace std;

namespace flex_neuralnet
{

class BaseTrainer
{
public:
   BaseTrainer(BaseNeuralNet& _net);
   virtual ~BaseTrainer();

   /* ***************************************************
    *    Public setter methods
    */
   void set_max_epochs(long _epochs);
   void set_global_learning_rate(double _rate);
   void set_layer_learning_rate(const string& _layerID, double _rate);
   void set_layer_biases_learning_rate(const string& _layerID, double _rate);
   void set_layer_weights_learning_rate(const string& _layerID, double _rate);
   void set_learning_momentum(double _rate);
   void set_performance_goal(double _perf);
   void set_min_gradient(double _min);
   void set_max_validation_failures(int _count);

   void set_batch_mode();
   void set_minibatch_mode(unsigned int _size);
   void set_online_mode();

   void set_random_order_mode(bool _val);
   void set_report_frequency(int _freq);
   void set_verbose(bool _mode);

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

   /* ****************************************************
    *    Virtual getter methods for network, error functor, and
    *    learning rate policies for each layer
    */
   virtual OutputErrorFunctor& get_errorfunc() = 0;
   virtual LearningRatePolicy& get_learning_rates() = 0;

   /* ****************************************************
    *    Training methods
    */
   virtual void train(const DataSet<Exemplar<Pattern,Pattern> >& trnset,
         const DataSet<Exemplar<Pattern,Pattern> >& vldset = DataSet<Exemplar<Pattern,Pattern> > (),
         const DataSet<Exemplar<Pattern,Pattern> >& tstset = DataSet<Exemplar<Pattern,Pattern> > ()) = 0;

   virtual void train(const DataSet<Exemplar<PatternSequence,PatternSequence> >& _trnset,
         const DataSet<Exemplar<PatternSequence,PatternSequence> >& _vldset = DataSet<Exemplar<PatternSequence,PatternSequence> > (),
         const DataSet<Exemplar<PatternSequence,PatternSequence> >& _tstset = DataSet<Exemplar<PatternSequence,PatternSequence> > ()) = 0;

   double global_perf();

   const TrainingRecord& training_record();

   /* ****************************************************
    *    Public training termination status
    */
   static const unsigned int TRAINING = 0;
   static const unsigned int MAX_EPOCHS_STOP = 1;
   static const unsigned int PERFORMANCE_GOAL_STOP = 2;
   static const unsigned int MIN_PERFORMANCE_GRADIENT_STOP = 3;
   static const unsigned int MAX_VALIDATION_FAILURES_STOP = 4;
   static const unsigned int MAX_CONSECUTIVE_FAILBACK_STOP = 5;

protected:

   /* ****************************************************
    *    Protected training helper methods
    */
   virtual int is_training_complete(const TrainingRecord::Entry& _trEntry, double& _prevValidPerf, int& _validFail);

   /* ****************************************************
    *    Protected methods to save and restore network
    *    network weights
    */
   void save_weights(const string& _id);
   void restore_weights(const string& _id);

   /* *****************************************************
    *    Protected helper methods for managing network
    *    weight training deltas and other weights cache
    */
   bool initialized;
   NetworkWeightsData& get_cached_network_weights(const string& _id);
   void alloc_network_weights_cache_entry(
         const string& _bufferID);
   void apply_delta_network_weights();
   void zero_delta_network_weights();

   virtual void alloc_network_learning_rates() = 0;


   /* *****************************************************
    *    Protected helper method to calculate the
    *    performance for a specified data set
    */
   template <class _InElem, class _TgtElem>
   double sim(const DataSet<Exemplar<_InElem,_TgtElem> >& dataset);

   double sim(const DataSet<Exemplar<PatternSequence,PatternSequence> >& trainset);

   void initialize_presentation_order(vector<unsigned int>& _order);
   void permute_presentation_order(vector<unsigned int>& _order);

   /* *******************************************************
    *    Protected methods to manage performance trace
    */
   TrainingRecord::Entry& calc_performance_data(unsigned int _epoch, const DataSet<Exemplar<Pattern,Pattern> >& trnset,
         const DataSet<Exemplar<Pattern,Pattern> >& vldset = DataSet<Exemplar<Pattern,Pattern> > (),
         const DataSet<Exemplar<Pattern,Pattern> >& tstset = DataSet<Exemplar<Pattern,Pattern> > ());

   bool is_best_perf(const TrainingRecord::Entry& _trEntry);

   void clear_training_record();
   void update_training_record(unsigned int _epoch, unsigned int _stopSig, const TrainingRecord::Entry& _trEntry);

   void print_training_record()
   {
      cout << "training record size " << last_training_record.size() << endl;
      for (unsigned int i=0; i<last_training_record.size(); i++)
      {
         TrainingRecord::Entry& entry = last_training_record.at(i);
         if (entry.contains_key(TrainingRecord::Entry::train_perf_id))
            cout << "training set perf " << entry.training_perf() << endl;
      }
   }

   /* *******************************************************
    *    Protected utility functions
    */
   int urand(int n);


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
    *    The network
    */
   BaseNeuralNet& neural_network;

   /* ************************************************
    *   Map of learning rates for network layers
    */
     LearningRatePolicy* network_learning_rates;

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

   /* ************************************************
    *    Performance data reporting parameters
    */
   int report_freq;

private:

   /* ***********************************************
    *   Working storage data members
    */
   map<string, NetworkWeightsData> network_weights_cache;

   /* ************************************************
    *    Training performance record
    */
   TrainingRecord last_training_record;
};

inline
void BaseTrainer::set_max_epochs(long _epochs)
{
   if (_epochs < 0)
   {
      ostringstream err_str;
      err_str << "BaseTrainer::set_max_epochs() Error : " << endl;
      err_str << "   Attempt to set max epochs to illegal value - " << _epochs << "." << endl;
      err_str << "   The value must be greater than or equal to zero." << endl;
      throw invalid_argument(err_str.str());
   }

   max_training_epochs = _epochs;
}

inline
void BaseTrainer::set_learning_momentum(double _rate)
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
void BaseTrainer::set_performance_goal(double _perf)
{
   perf_goal = _perf;
}

inline
void BaseTrainer::set_min_gradient(double _min)
{
   min_perf_gradient = _min;
}

inline
void BaseTrainer::set_max_validation_failures(int _count)
{
   max_validation_fail = _count;
}

inline
void BaseTrainer::set_batch_mode()
{
   batch_mode = true;
   batch_size = 0;
}

inline
void BaseTrainer::set_minibatch_mode(unsigned int _size)
{
   batch_mode = true;
   batch_size = _size;
}

inline
void BaseTrainer::set_online_mode()
{
   batch_mode = false;
}

inline
void BaseTrainer::set_random_order_mode(bool _val)
{
   random_order_mode = _val;
}

inline
void BaseTrainer::set_verbose(bool _mode)
{
   verbose_mode = _mode;
}

inline
void BaseTrainer::set_report_frequency(int _freq)
{
   report_freq = _freq;
}

inline
long BaseTrainer::max_epochs()
{
   return max_training_epochs;
}

inline
double BaseTrainer::performance_goal()
{
   return perf_goal;
}

inline
double BaseTrainer::min_gradient()
{
   return min_perf_gradient;
}

inline
int BaseTrainer::max_validation_failures()
{
   return max_validation_fail;
}

inline
bool BaseTrainer::is_batch_mode()
{
   return batch_mode && (batch_size == 0);
}

inline
bool BaseTrainer::is_minibatch_mode()
{
   return batch_mode && (batch_size > 0);
}

inline
bool BaseTrainer::is_online_mode()
{
   return !batch_mode;
}

inline
unsigned int BaseTrainer::minibatch_size()
{
   return batch_size;
}

inline
bool BaseTrainer::is_random_order_mode()
{
   return random_order_mode;
}

inline
bool BaseTrainer::is_verbose_mode()
{
   return verbose_mode;
}

inline
int BaseTrainer::report_frequency()
{
   return report_freq;
}

inline
double BaseTrainer::global_perf()
{
   return last_training_record.best_training_perf();
}

inline
const TrainingRecord& BaseTrainer::training_record()
{
   return last_training_record;
}

inline
void BaseTrainer::clear_training_record()
{
   last_training_record.clear();
}

template <class _InElem, class _TgtElem>
double BaseTrainer::sim(const DataSet<Exemplar<_InElem,_TgtElem> >& trainset)
{
   long epoch_no;
   unsigned int trainset_ndx;
   double perf;

   OutputErrorFunctor& error_func = get_errorfunc();
   double global_performance = 0;

   for (trainset_ndx = 0; trainset_ndx < trainset.size(); trainset_ndx++)
   {
      const Exemplar<_InElem,_TgtElem>& exemplar = trainset.at(trainset_ndx);

      _InElem rawin = exemplar.input();
      _TgtElem tgtout = exemplar.target_output();

      // Present input pattern and get output
      const _TgtElem& network_output = neural_network(rawin);

      // Calculate the output error
      double isse;
      vector<double> gradient(network_output().size());
      error_func(isse, gradient, network_output, tgtout);

      global_performance += isse;

   }
   global_performance /= trainset.size();

   return global_performance;
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_BASETRAINER_H_ */
