//
// Created by kfedrick on 3/20/21.
//

#ifndef FLEX_NEURALNET_NEWBASETRAINING_H_
#define FLEX_NEURALNET_NEWBASETRAINING_H_

#include <vector>
#include <map>
#include <flexnnet.h>

namespace flexnnet
{
   class BaseTrainer
   {
      struct CachedValues
      {
         // Cached layer weights (e.g. used to back out weight changes)
         std::map<std::string, NetworkWeights> layer_weights;

         // Cumulative NN weight updates prior to application
         std::map<std::string, Array2D<double>> cumulative_weight_updates;
      };

      using NetworkWeights = std::map<std::string, LayerWeights>;

   protected:

      BaseTrainer();

      BaseTrainer(const BaseTrainer& _nnet);

      CachedValues& alloc(const BaseNeuralNet& _nnet);

      void copy(const BaseTrainer& _tnnet);

   public:
      BaseTrainer& operator=(const BaseTrainer& _nnet);

      const TrainingReport& get_training_report(void) const;

      /**
       * Calculate the neural network weight updates given the specified
       * output error gradient.
       *
       * Precondition:
       *    The network layers retain the state information
       *    for the input/target training exemplar used to generate
       *    the error gradient.
       *
       * @param _egradient
       */
      virtual void calc_weight_updates(const ValarrMap _egradient) = 0;

   protected:

      void save_training_record(const TrainingRecord& _trec);

      /**
       * Save a copy of the neural network weights to the cache using
       * using the specified label.
       *
       * @param _id
       */
      void save_network_weights(const BaseNeuralNet& _nnet, const std::string& _id);

      /**
       * Restore the neural network weights from the cached values specified
       * by the label.
       *
       * @param _id
       */
      void restore_network_weights(BaseNeuralNet& _nnet, const std::string& _id);

      void adjust_network_weights(BaseNeuralNet& _nnet);

      void accumulate_weight_updates(const BaseNeuralNet& _nnet, const std::string& _id, const Array2D<double>& _deltaw);

   private:
      CachedValues& get_cache(const BaseNeuralNet& _nnet);

   private:
      // List of best basic_layer weights
      TrainingReport training_report;

      std::map<const BaseNeuralNet*, CachedValues> cached_nn_values;
   };

   inline
   BaseTrainer::BaseTrainer()
   {
   }

   inline
   BaseTrainer::BaseTrainer(const BaseTrainer& _nnet)
   {
      copy(_nnet);
   }

   BaseTrainer& BaseTrainer::operator=(const BaseTrainer& _nnet)
   {
      copy(_nnet);
      return *this;
   }

   inline
   const TrainingReport& BaseTrainer::get_training_report(void) const
   {
      return training_report;
   }

   inline
   void BaseTrainer::save_training_record(const TrainingRecord& _trec)
   {
      training_report.add_record(_trec);
   }

   inline
   void BaseTrainer::copy(const BaseTrainer& _tnnet)
   {
      cached_nn_values = _tnnet.cached_nn_values;
   }

   inline
   BaseTrainer::CachedValues& BaseTrainer::alloc(const BaseNeuralNet& _nnet)
   {
      CachedValues& cache = cached_nn_values[&_nnet] = CachedValues();

      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = _nnet.get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;
         const LayerWeights& w = it.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         // Create a cumulative weight update entry for this layer
         cache.cumulative_weight_updates[id] = {};
         cache.cumulative_weight_updates[id].resize(dim.rows, dim.cols);
      }

      return cache;
   }

   inline
   BaseTrainer::CachedValues& BaseTrainer::get_cache(const BaseNeuralNet& _nnet)
   {
      std::map<const BaseNeuralNet*, CachedValues>::iterator entry = cached_nn_values.find(&_nnet);

      if (entry != cached_nn_values.end())
         return entry->second;

      alloc(_nnet);
      return cached_nn_values[&_nnet];
   }

   inline
   void BaseTrainer::save_network_weights(const BaseNeuralNet& _nnet, const std::string& _id)
   {
      CachedValues& cache = get_cache(_nnet);

      NetworkWeights network_weights = cache.layer_weights[_id];
      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = _nnet.get_layers();

      for (auto& it : layers)
         network_weights[it.first] = it.second->weights();
   }

   inline
   void BaseTrainer::restore_network_weights(BaseNeuralNet& _nnet, const std::string& _id)
   {
      CachedValues& cache = get_cache(_nnet);

      NetworkWeights network_weights = cache.layer_weights[_id];
      for (auto& it : network_weights)
         _nnet.set_weights(it.first, it.second);
   }

   inline
   void BaseTrainer::adjust_network_weights(BaseNeuralNet& _nnet)
   {
      CachedValues& cache = get_cache(_nnet);

      for (auto& it : cache.cumulative_weight_updates)
      {
         _nnet.adjust_weights(it.first, it.second);
         it.second = 0;
      }
   }

   inline
   void BaseTrainer::accumulate_weight_updates(const BaseNeuralNet& _nnet, const std::string& _id, const Array2D<double>& _deltaw)
   {
      CachedValues& cache = get_cache(_nnet);

      cache.cumulative_weight_updates[_id] += _deltaw;
   }

} // end namespace flexnnet

#endif //FLEX_NEURALNET_BASETRAINING_H_
