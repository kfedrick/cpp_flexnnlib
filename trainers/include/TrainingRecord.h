//
// Created by kfedrick on 10/5/19.
//

#ifndef FLEX_NEURALNET_TRAININGRECORD_H_
#define FLEX_NEURALNET_TRAININGRECORD_H_

#include <cstddef>
#include <vector>

#include "flexnnet_trainers.h"

namespace flexnnet
{
   struct TrainingRecordEntry
   {
      size_t epoch;
      double performance;
   };

   struct TrainingRecord
   {
      size_t best_epoch;
      double best_performance;
      TrainingStopSignal stop_signal;

      std::vector<TrainingRecordEntry> training_set_trace;
      std::vector<TrainingRecordEntry> validation_set_trace;
      std::vector<TrainingRecordEntry> test_set_trace;

      void clear(void);
   };

   void TrainingRecord::clear(void)
   {
      training_set_trace.clear();
      validation_set_trace.clear();
      test_set_trace.clear();
   }
}

#endif //FLEX_NEURALNET_TRAININGRECORD_H_
