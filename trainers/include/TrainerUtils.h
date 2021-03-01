//
// Created by kfedrick on 9/21/19.
//

#ifndef FLEX_NEURALNET_TRAINERUTILS_H_
#define FLEX_NEURALNET_TRAINERUTILS_H_

#include <cstddef>
#include <vector>
#include <iostream>

#include "flexnnet.h"

#include "TrainingRecord.h"
#include "BasicNeuralNet.h"

namespace flexnnet
{
   struct TrainingStatistics
   {
      double mean_perf;
      double stddev_perf;
      std::vector<double> raw_perf_histogram;
   };

   using NetworkWeights = std::map<std::string, LayerWeights>;

   struct TrainedNNetRecord
   {
      NetworkWeights network_weights;
      TrainingRecord training_record;

      bool operator<(const TrainedNNetRecord& _rec) const
      {
         return training_record.best_performance < _rec.training_record.best_performance;
      }
   };

   class TrainerUtils
   {
   private:
      struct NetworkWeightsRecord
      {
         double performance;
         NetworkWeights layer_weights;

         bool operator<(const NetworkWeightsRecord& _rec) const
         {
            return performance < _rec.performance;
         }
      };

   public:
      const TrainingStatistics& get_training_statistics(void) const;
      const std::vector<TrainedNNetRecord>& get_best_networks(void) const;

   protected:

      void save_nnet_weights(const std::string& _label, const BasicNeuralNet& _nnet);
      void restore_nnet_weights(const std::string& _label, BasicNeuralNet& _nnet);
      void save_best_weights(TrainingRecord& _trec, const BasicNeuralNet& _nnet);

   private:
      TrainingStatistics training_stats;

      // Cached basic_layer weights (e.g. used to back out weight changes)
      std::map<std::string, NetworkWeights> cached_layer_weights;

      // List of best basic_layer weights
      std::vector<TrainedNNetRecord> best_nnets;
   };

   void TrainerUtils::save_nnet_weights(const std::string& _label, const BasicNeuralNet& _nnet)
   {
      std::cout << "TrainerUtils::save_nnet_weights() - entry\n";
      cached_layer_weights.emplace(_label, _nnet.get_weights());
   }

   inline
   void TrainerUtils::restore_nnet_weights(const std::string& _label, BasicNeuralNet& _nnet)
   {
      std::cout << "TrainerUtils::restore_nnet_weights() - entry\n";

      NetworkWeights network_weights = cached_layer_weights[_label];
      _nnet.set_weights(network_weights);
   }

   inline
   void TrainerUtils::save_best_weights(TrainingRecord& _trec, const BasicNeuralNet& _nnet)
   {
      TrainedNNetRecord nnrec;
      nnrec.training_record = _trec;
      nnrec.network_weights = _nnet.get_weights();

      best_nnets.push_back(nnrec);
      std::sort(best_nnets.begin(), best_nnets.end());

      best_nnets.resize(10);
   }

   inline
   const std::vector<TrainedNNetRecord>& TrainerUtils::get_best_networks(void) const
   {
      return best_nnets;
   }

   inline
   const TrainingStatistics& TrainerUtils::get_training_statistics(void) const
   {
      return training_stats;
   }

}

#endif //FLEX_NEURALNET_TRAINERUTILS_H_
