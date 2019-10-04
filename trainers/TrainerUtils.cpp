//
// Created by kfedrick on 9/21/19.
//

#include "TrainerUtils.h"

#include <sstream>

using flexnnet::TrainerUtils;
using flexnnet::NNetTrainingStatistics;

TrainerUtils::TrainerUtils(void)
{
   training_runs = DEFAULT_TRAINING_RUNS;
   saved_nnet_limit = DEFAULT_SAVED_NNET_LIMIT;
}

void TrainerUtils::collect_training_stats (const TrainingRecord &_tr)
{
   training_stats.raw_trainingset_perf.push_back (_tr.final_trainingset_performance ());
   training_stats.raw_validset_perf.push_back (_tr.final_validationset_performance ());
   training_stats.raw_testset_perf.push_back (_tr.final_testset_performance ());
}

const NNetTrainingStatistics &TrainerUtils::get_training_statistics (void) const
{
   return training_stats;
}

void TrainerUtils::set_training_runs (unsigned int _count)
{
   if (_count < 1)
   {
      std::ostringstream err_str;
      err_str
         << "Error : BasicTrainer.set_training_runs() - invalid value " << _count << " for training runs.\n";
      throw std::invalid_argument (err_str.str ());
   }

   training_runs = _count;
}

void TrainerUtils::set_saved_network_limit (unsigned int _count)
{
   if (_count < 1)
   {
      std::ostringstream err_str;
      err_str
         << "Error : BasicTrainer.set_saved_network_limit() - invalid value " << _count << " for training runs.\n";
      throw std::invalid_argument (err_str.str ());
   }

   saved_nnet_limit = _count;
}

