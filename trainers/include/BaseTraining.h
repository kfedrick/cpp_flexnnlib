//
// Created by kfedrick on 3/20/21.
//

#ifndef FLEX_NEURALNET_BASETRAINING_H_
#define FLEX_NEURALNET_BASETRAINING_H_

#include <vector>
#include <map>
#include <flexnnet.h>
#include <BaseNeuralNet.h>
#include "TrainerConfig.h"
#include "TrainingRecord.h"
#include <TrainingReport.h>

namespace flexnnet
{
   class BaseTraining : public TrainerConfig
   {
      using NetworkWeights = std::map<std::string, LayerWeights>;

   public:
      const TrainingReport& get_training_report(void) const;

   protected:
      void initialize(BaseNeuralNet& _nnet);

      /**
       * Save a copy of the neural network weights to the cache using
       * using the specified label.
       *
       * @param _label
       * @param _nnet
       */
      void save_nnet_weights(const std::string& _label, const BaseNeuralNet& _nnet);

      /**
       * Restore the neural network weights from the cached values specified
       * by the label.
       *
       * @param _label
       * @param _nnet
       */
      void restore_nnet_weights(const std::string& _label, BaseNeuralNet& _nnet);


      void save_best_weights(const TrainingRecord& _trec, const BaseNeuralNet& _nnet);

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
       * @param _nnet
       */
      virtual void calc_weight_updates(const ValarrMap _egradient, BaseNeuralNet& _nnet) = 0;

      void adjust_network_weights(BaseNeuralNet& _nnet);

      void accumulate_weight_updates(const std::map<std::string, Array2D<double>>& _deltaw);

   private:
      // Cached basic_layer weights (e.g. used to back out weight changes)
      std::map<std::string, NetworkWeights> cached_layer_weights;

      // Cumulative NN weight updates prior to application
      std::map<std::string, Array2D<double>> cumulative_weight_updates;

      // List of best basic_layer weights
      TrainingReport training_report;
   };

   inline
   void BaseTraining::initialize(BaseNeuralNet& _nnet)
   {
      training_report.set_max_records(saved_nnet_limit());

      cumulative_weight_updates.clear();

      std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = _nnet.get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;
         LayerWeights& w = it.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         // Create a cumulative weight update entry for this layer
         cumulative_weight_updates[id] = {};
         cumulative_weight_updates[id].resize(dim.rows, dim.cols);
      }
   }

   inline
   const TrainingReport& BaseTraining::get_training_report(void) const
   {
      return training_report;
   }

   inline
   void BaseTraining::save_best_weights(const TrainingRecord& _trec, const BaseNeuralNet& _nnet)
   {
      training_report.add_record(_trec);
   }

   inline
   void BaseTraining::save_nnet_weights(const std::string& _label, const BaseNeuralNet& _nnet)
   {
      cached_layer_weights[_label] = _nnet.get_weights();
   }

   inline
   void BaseTraining::restore_nnet_weights(const std::string& _label, BaseNeuralNet& _nnet)
   {
      NetworkWeights network_weights = cached_layer_weights[_label];
      _nnet.set_weights(network_weights);
   }

   inline
   void BaseTraining::adjust_network_weights(BaseNeuralNet& _nnet)
   {
      for (auto& it : cumulative_weight_updates)
      {
         _nnet.adjust_weights(it.first, it.second);
         it.second = 0;
      }
   }

   inline
   void BaseTraining::accumulate_weight_updates(const std::map<std::string, Array2D<double>>& _deltaw)
   {
      for (auto& it : cumulative_weight_updates)
         it.second += _deltaw.at(it.first);
   }

}

#endif //FLEX_NEURALNET_BASETRAINING_H_
