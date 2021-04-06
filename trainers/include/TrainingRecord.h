//
// Created by kfedrick on 10/5/19.
//

#ifndef FLEX_NEURALNET_TRAININGRECORD_H_
#define FLEX_NEURALNET_TRAININGRECORD_H_

//#include <cstddef>
#include <vector>
#include <flexnnet.h>
#include <flexnnet_trainers.h>

namespace flexnnet
{
   struct TrainingRecordEntry
   {
      size_t epoch;
      double performance;
   };

   struct TrainingRecord
   {
      /*
       * Training run info and performance statistics
       */
      size_t best_epoch;
      double best_performance;
      TrainingStopSignal stop_signal;
      NetworkWeights network_weights;

      /*
       * Training run performance traces
       */
      std::vector<TrainingRecordEntry> training_set_trace;
      std::vector<TrainingRecordEntry> validation_set_trace;
      std::vector<TrainingRecordEntry> test_set_trace;

      bool operator<(const TrainingRecord& _rec) const;
      void clear(void);
   };

   inline
   bool TrainingRecord::operator<(const TrainingRecord& _rec) const
   {
      /*
       * Order records based on the lowest best_performance.
       * To keep items from evaluating as equal, break ties by
       * looking at the best epoch. If the best epoch is equal
       * chose the lowest valued memory address.
       */
      if (best_performance == _rec.best_performance)
         if (best_epoch == _rec.best_epoch)
            return this < &_rec;
         else
            return best_epoch < _rec.best_epoch;
      else
         return best_performance < _rec.best_performance;
   }

   inline
   void TrainingRecord::clear(void)
   {
      training_set_trace.clear();
      validation_set_trace.clear();
      test_set_trace.clear();
   }
}

#endif //FLEX_NEURALNET_TRAININGRECORD_H_
