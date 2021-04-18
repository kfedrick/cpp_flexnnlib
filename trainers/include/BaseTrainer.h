//
// Created by kfedrick on 3/20/21.
//

#ifndef FLEX_NEURALNET_NEWBASETRAINING_H_
#define FLEX_NEURALNET_NEWBASETRAINING_H_

#include <vector>
#include <map>
#include <flexnnet.h>
#include <BaseNeuralNet.h>
#include "TrainerConfig.h"
#include "TrainingRecord.h"
#include <TrainingReport.h>

namespace flexnnet
{
   template<class InTyp, class OutTyp, template<class, class> class NN>
   class BaseTrainer : public TrainerConfig
   {
      using NNTyp = NN<InTyp, OutTyp>;
      using NetworkWeights = std::map<std::string, LayerWeights>;

   public:
      BaseTrainer(NNTyp& _nnet);
      const TrainingReport& get_training_report(void) const;

   protected:
      void alloc();

      /**
       * Save a copy of the neural network weights to the cache using
       * using the specified label.
       *
       * @param _label
       */
      void save_network_weights(const std::string& _label);

      /**
       * Restore the neural network weights from the cached values specified
       * by the label.
       *
       * @param _label
       */
      void restore_network_weights(const std::string& _label);

      void save_training_record(const TrainingRecord& _trec);

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

      void adjust_network_weights();

      void accumulate_weight_updates(const std::string& _id, const Array2D<double>& _deltaw);

   protected:
      NNTyp& nnet;

   private:
      // Cached basic_layer weights (e.g. used to back out weight changes)
      std::map<std::string, NetworkWeights> cached_layer_weights;

      // Cumulative NN weight updates prior to application
      std::map<std::string, Array2D<double>> cumulative_weight_updates;

      // List of best basic_layer weights
      TrainingReport training_report;
   };

   template<class InTyp, class OutTyp, template<class, class> class NN>
   inline
   BaseTrainer<InTyp, OutTyp, NN>::BaseTrainer(NNTyp& _nnet) : nnet(_nnet), TrainerConfig()
   {
   }

   template<class InTyp, class OutTyp, template<class, class> class NN>
   inline
   void BaseTrainer<InTyp, OutTyp, NN>::alloc()
   {
      training_report.set_max_records(saved_nnet_limit());

      cumulative_weight_updates.clear();

      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = nnet.get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;
         const LayerWeights& w = it.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         // Create a cumulative weight update entry for this layer
         cumulative_weight_updates[id] = {};
         cumulative_weight_updates[id].resize(dim.rows, dim.cols);
      }
   }

   template<class InTyp, class OutTyp, template<class, class> class NN>
   inline
   const TrainingReport& BaseTrainer<InTyp, OutTyp, NN>::get_training_report(void) const
   {
      return training_report;
   }

   template<class InTyp, class OutTyp, template<class, class> class NN>
   inline
   void BaseTrainer<InTyp, OutTyp, NN>::save_training_record(const TrainingRecord& _trec)
   {
      training_report.add_record(_trec);
   }

   template<class InTyp, class OutTyp, template<class, class> class NN>
   inline
   void BaseTrainer<InTyp, OutTyp, NN>::save_network_weights(const std::string& _label)
   {
      NetworkWeights network_weights = cached_layer_weights[_label];
      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = nnet.get_layers();

      for (auto& it : layers)
         network_weights[it.first] = it.second->weights();
   }

   template<class InTyp, class OutTyp, template<class, class> class NN>
   inline
   void BaseTrainer<InTyp, OutTyp, NN>::restore_network_weights(const std::string& _label)
   {
      NetworkWeights network_weights = cached_layer_weights[_label];
      for (auto& it : network_weights)
         nnet.set_weights(it.first, it.second);
   }

   template<class InTyp, class OutTyp, template<class, class> class NN>
   inline
   void BaseTrainer<InTyp, OutTyp, NN>::adjust_network_weights()
   {
      for (auto& it : cumulative_weight_updates)
      {
         nnet.adjust_weights(it.first, it.second);
         it.second = 0;
      }
   }

   template<class InTyp, class OutTyp, template<class, class> class NN>
   inline
   void BaseTrainer<InTyp, OutTyp, NN>::accumulate_weight_updates(const std::string& _id, const Array2D<double>& _deltaw)
   {
      cumulative_weight_updates[_id] += _deltaw;
   }

}

#endif //FLEX_NEURALNET_BASETRAINING_H_
