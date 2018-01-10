/*
 * BaseTrainer.cpp
 *
 *  Created on: Mar 27, 2014
 *      Author: kfedrick
 */

#include "BaseTrainer.h"

namespace flex_neuralnet
{

const long BaseTrainer::default_max_epochs = 1;
const double BaseTrainer::default_learning_momentum = 0.0;
const double BaseTrainer::default_performance_goal = 0.0;
const double BaseTrainer::default_min_gradient = 0.0;
const int BaseTrainer::default_max_valid_fail = 10;
const bool BaseTrainer::default_batch_mode = true;
const unsigned int BaseTrainer::default_batch_size = 0;
const int BaseTrainer::default_report_frequency = 1;

const string BaseTrainer::delta_network_weights_id = "delta_network_weights";
const string BaseTrainer::previous_delta_network_weights_id =
      "previous_delta_network_weights";
const string BaseTrainer::adjusted_network_weights_id =
      "adjusted_network_weights";
const string BaseTrainer::best_weights_id = "best_weights";
const string BaseTrainer::failback_weights_id = "failback_weights";

BaseTrainer::BaseTrainer(BaseNeuralNet& _net) :
      neural_network(_net)
{
   srand(time(NULL));

   network_learning_rates = NULL;

   set_max_epochs(default_max_epochs);
   set_learning_momentum(default_learning_momentum);
   set_performance_goal(default_performance_goal);
   set_min_gradient(default_min_gradient);
   set_max_validation_failures(default_max_valid_fail);
   set_batch_mode();
   set_report_frequency(default_report_frequency);

   initialized = false;
}

BaseTrainer::~BaseTrainer()
{
   // TODO Auto-generated destructor stub
}

NetworkWeightsData& BaseTrainer::get_cached_network_weights(const string& _id)
{
   if (network_weights_cache.find(_id) == network_weights_cache.end())
      alloc_network_weights_cache_entry(_id);

   return network_weights_cache[_id];
}

void BaseTrainer::set_global_learning_rate(double _rate)
{
   alloc_network_learning_rates();
   network_learning_rates->set_global_learning_rate(_rate);
}

void BaseTrainer::set_layer_learning_rate(const string& _layerID, double _rate)
{
   alloc_network_learning_rates();
   network_learning_rates->set_layer_learning_rate(_layerID, _rate);
}

void BaseTrainer::set_layer_biases_learning_rate(const string& _layerID, double _rate)
{
   alloc_network_learning_rates();
   network_learning_rates->set_layer_biases_learning_rate(_layerID, _rate);
}

void BaseTrainer::set_layer_weights_learning_rate(const string& _layerID, double _rate)
{
   alloc_network_learning_rates();
   network_learning_rates->set_layer_weights_learning_rate(_layerID, _rate);
}

double BaseTrainer::sim(
      const DataSet<Exemplar<PatternSequence, PatternSequence> >& dataset)
{
   unsigned int trainset_ndx;

   double isse;
   double sequence_performance = 0;
   double global_performance = 0;

   OutputErrorFunctor& error_func = get_errorfunc();

   for (trainset_ndx = 0; trainset_ndx < dataset.size(); trainset_ndx++)
   {
      const Exemplar<PatternSequence, PatternSequence>& exemplar = dataset.at(
            trainset_ndx);

      PatternSequence rawin = exemplar.input();
      PatternSequence tgtoutseq = exemplar.target_output();

      // Present input pattern and get output
      const PatternSequence& network_output_seq = neural_network(rawin);

      // Calculate the output error

      for (unsigned int pattern_ndx = 0;
            pattern_ndx < network_output_seq.size(); pattern_ndx++)
      {
         const Pattern& tgtpatt = tgtoutseq[pattern_ndx];
         if (tgtpatt.size() == 0)
            continue;

         const Pattern& network_output = network_output_seq.at(pattern_ndx);
         vector<double> gradient(network_output.size());
         error_func(isse, gradient, network_output, tgtpatt);

         sequence_performance += isse;
      }
      sequence_performance /= network_output_seq.size();

      global_performance += sequence_performance;
   }
   global_performance /= (dataset.size());

   return global_performance;
}

int BaseTrainer::is_training_complete(const TrainingRecord::Entry& _trEntry,
      double& _prevValidPerf, int& _validFail)
{
   if (_trEntry.epoch() >= max_training_epochs)
      return MAX_EPOCHS_STOP;

   if (_trEntry.training_perf() < perf_goal)
      return PERFORMANCE_GOAL_STOP;

   // Check for validation set increasing if there is one
   if (_trEntry.contains_key(TrainingRecord::Entry::valid_perf_id))
   {
      if (_trEntry.validation_perf() > _prevValidPerf)
         _validFail++;
      else
         _validFail = 0;

      if (_validFail > max_validation_fail)
         return MAX_VALIDATION_FAILURES_STOP;
   }

   // Check training gradient if it's available
   if (_trEntry.contains_key(TrainingRecord::Entry::train_grad_id))
      if (_trEntry.training_gradient() < min_perf_gradient)
         return MIN_PERFORMANCE_GRADIENT_STOP;

   return TRAINING;
}

void BaseTrainer::save_weights(const string& _id)
{
   // TODO - check the buffer id to make sure it's not one of the reserved values

   NetworkWeightsData& network_weights_data = get_cached_network_weights(_id);
   /*
    // Initialize weights if this is first save
    if (network_weights_cache.find(_id) == network_weights_cache.end())
    {
    NetworkWeightsData& network_weights_data = network_weights_cache[_id];
    network_weights_data.fromNet(neuralnet);
    }

    NetworkWeightsData& network_weights_data = network_weights_cache[_id];
    */

   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      LayerWeightsData& layer_weights_data = network_weights_data.layer_weights(
            name);

      // Layer value at time step 0 is the initial layer value
      layer_weights_data.initial_value = layer(0);

      layer_weights_data.biases = layer.get_biases();
      layer_weights_data.weights = layer.get_weights();
   }
}

void BaseTrainer::restore_weights(const string& _id)
{
   // TODO - check the buffer id to make sure it's not one of the reserved values

   NetworkWeightsData& network_weights_data = network_weights_cache[_id];

   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      LayerWeightsData& layer_weights_data = network_weights_data.layer_weights(
            name);

      layer.set_biases(layer_weights_data.biases);
      layer.set_weights(layer_weights_data.weights);
   }
}

void BaseTrainer::alloc_network_weights_cache_entry(const string& _id)
{
   NetworkWeightsData& network_weights = network_weights_cache[_id];

   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      const string& name = layer.name();

      const Array<double>& layer_weights = layer.get_weights();
      unsigned int rows = layer_weights.rowDim();
      unsigned int cols = layer_weights.colDim();

      LayerWeightsData& layer_weights_data = network_weights.new_layer_weights(
            name);

      layer_weights_data.initial_value.resize(layer.size(), 0.0);
      layer_weights_data.biases.resize(layer.get_biases().size(), 0.0);

      layer_weights_data.weights.resize(rows, cols);
      layer_weights_data.weights = 0;
   }
}

void BaseTrainer::apply_delta_network_weights()
{
   //cout << " *** apply delta weights *****" << endl;

   NetworkWeightsData& delta_network_weights = get_cached_network_weights(
         delta_network_weights_id);
   NetworkWeightsData& prev_delta_network_weights = get_cached_network_weights(
         previous_delta_network_weights_id);
   NetworkWeightsData& adjusted_network_weights = get_cached_network_weights(
         adjusted_network_weights_id);

   const vector<BaseLayer*> network_layers =
         neural_network.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

         //cout << "============= " << layer.name()
         //      << " ==============" << endl;

      const LayerWeightsData& delta_layer_weights_data =
            delta_network_weights.layer_weights(name);
      const LayerWeightsData& prev_delta_layer_weights_data =
            prev_delta_network_weights.layer_weights(name);
      LayerWeightsData& adjusted_layer_weights_data =
            adjusted_network_weights.layer_weights(name);

      const vector<double>& delta_layer_biases = delta_layer_weights_data.biases;
      const vector<double>& prev_delta_layer_biases =
            prev_delta_layer_weights_data.biases;
      vector<double>& adjusted_layer_biases = adjusted_layer_weights_data.biases;

      //cout << "   biases : " << endl;

      adjusted_layer_biases = layer.get_biases();
      for (unsigned int ndx = 0; ndx < adjusted_layer_biases.size(); ndx++)
      {
         adjusted_layer_biases.at(ndx) += learning_momentum
               * prev_delta_layer_biases.at(ndx) + delta_layer_biases.at(ndx);

         //if (ndx > 0)
          //  cout << ", ";
         //cout << delta_layer_biases.at(ndx);
      }
      //cout << endl << "-------------" << endl;
      if (layer.is_learn_biases())
         layer.set_biases(adjusted_layer_biases);

      const Array<double>& delta_layer_weights =
            delta_layer_weights_data.weights;
      const Array<double>& prev_delta_layer_weights =
            prev_delta_layer_weights_data.weights;
      Array<double>& adjusted_layer_weights =
            adjusted_layer_weights_data.weights;

      adjusted_layer_weights = layer.get_weights();

      unsigned int row_sz = adjusted_layer_weights.rowDim();
      unsigned int col_sz = adjusted_layer_weights.colDim();

      //cout << "   weights : " << endl;

      for (unsigned int row = 0; row < row_sz; row++)
      {
         for (unsigned int col = 0; col < col_sz; col++)
         {
            adjusted_layer_weights.at(row, col) += learning_momentum
                  * prev_delta_layer_weights.at(row, col)
                  + delta_layer_weights.at(row, col);

            //if (col > 0)
            //   cout << ", ";
            //cout << delta_layer_weights.at(row, col);
         }
         //cout << endl;
      }

      if (layer.is_learn_weights())
         layer.set_weights(adjusted_layer_weights);
   }
   //cout << "-------------" << endl;

   // TODO - decide if we should save off the current delta weights here
   prev_delta_network_weights = delta_network_weights;
}

void BaseTrainer::zero_delta_network_weights()
{
   // Get a reference from the buffer to the entry containing the delta network weights
   NetworkWeightsData& delta_network_weights = get_cached_network_weights(
         delta_network_weights_id);

   // Iterate through the weights for each layer

   const set<string>& key_set = delta_network_weights.keySet();
   set<string>::iterator iter;
   for (iter = key_set.begin(); iter != key_set.end(); iter++)
   {
      const string& key_str = *iter;
      LayerWeightsData& layer_weights_data =
            delta_network_weights.layer_weights(key_str);

      unsigned int sz;

      // Clear deltas for initial layer value
      sz = layer_weights_data.initial_value.size();
      for (unsigned int ndx = 0; ndx < sz; ndx++)
         layer_weights_data.initial_value[ndx] = 0;

      // Clear deltas for layer biases
      sz = layer_weights_data.biases.size();
      for (unsigned int ndx = 0; ndx < sz; ndx++)
         layer_weights_data.biases[ndx] = 0;

      // Clear deltas for layer weights
      layer_weights_data.weights = 0;
   }
}

TrainingRecord::Entry& BaseTrainer::calc_performance_data(unsigned int _epoch,
      const DataSet<Exemplar<Pattern, Pattern> >& _trnset,
      const DataSet<Exemplar<Pattern, Pattern> >& _vldset,
      const DataSet<Exemplar<Pattern, Pattern> >& _tstset)
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

bool BaseTrainer::is_best_perf(const TrainingRecord::Entry& _trEntry)
{
   /*
    * If there's a validation set then check to see if the current validation
    * set performance is better than the previous best. Otherwise check the
    * performance on the training set.
    */
   if (_trEntry.contains_key(TrainingRecord::Entry::valid_perf_id))
   {
      if (_trEntry.validation_perf()
            < last_training_record.best_validation_perf())
         return true;
   }
   else
   {
      if (_trEntry.training_perf() < last_training_record.best_training_perf())
         return true;
   }

   return false;
}

void BaseTrainer::update_training_record(unsigned int _epoch,
      unsigned int _stopSig, const TrainingRecord::Entry& _trnrecEntry)
{
   /*
    * Store the new common information
    */
   last_training_record.set_training_epochs(_epoch);

   bool new_best_flag = false;

   /*
    * If the training performance is better than the previous best then
    * update the best training set performance data in the training record
    */
   if (_trnrecEntry.training_perf() < last_training_record.best_training_perf())
   {
      last_training_record.set_best_training_epoch(_epoch,
            _trnrecEntry.training_perf());
   }

   /*
    * If there's a validation set and the validation performance is better
    * than the previous value then update the best validation set performance
    * data in the training record
    */
   if (_trnrecEntry.contains_key(TrainingRecord::Entry::valid_perf_id))
   {
      if (_trnrecEntry.validation_perf()
            < last_training_record.best_validation_perf())
      {
         new_best_flag = true;
         last_training_record.set_best_validation_epoch(_epoch,
               _trnrecEntry.validation_perf());
      }
   }

   last_training_record.set_stop_signal(_stopSig);

   /*
    * Store the trace information for this epoch at intervals specified
    * by report frequency,
    * or this is the first training epoch (after training)
    * or if training has been terminated,
    * or if a new best performance was reached
    *
   if (_epoch < 10 || _epoch % report_freq == 0
         || _stopSig != BaseTrainer::TRAINING || new_best_flag)
      */
   if (_epoch < 10 || _epoch % report_freq == 0
            || _stopSig != BaseTrainer::TRAINING)
      last_training_record.push_back(_trnrecEntry);
}

void BaseTrainer::initialize_presentation_order(vector<unsigned int>& _ordervec)
{
   for (unsigned int ndx = 0; ndx < _ordervec.size(); ndx++)
      _ordervec[ndx] = ndx;

   if (random_order_mode)
      permute_presentation_order(_ordervec);
}

int BaseTrainer::urand(int n)
{
   int top = ((((RAND_MAX - n) + 1) / n) * n - 1) + n;
   int r;
   do
   {
      r = rand();
   } while (r > top);
   return (r % n);
}

void BaseTrainer::permute_presentation_order(vector<unsigned int>& _ordervec)
{
   unsigned int temp, new_ndx;
   unsigned int sz = _ordervec.size();

   for (unsigned int rounds = 0; rounds < 2; rounds++)
   {
      for (unsigned int ndx = 0; ndx < _ordervec.size(); ndx++)
      {
         new_ndx = urand(sz);

         temp = _ordervec[new_ndx];
         _ordervec[new_ndx] = _ordervec[ndx];
         _ordervec[ndx] = temp;
      }
   }
}

} /* namespace flex_neuralnet */
