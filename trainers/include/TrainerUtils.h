//
// Created by kfedrick on 9/21/19.
//

#ifndef FLEX_NEURALNET_TRAINERUTILS_H_
#define FLEX_NEURALNET_TRAINERUTILS_H_

#include <cstddef>
#include <vector>
#include <iostream>

#include "flexneuralnet.h"

#include "TrainingRecord.h"

namespace flexnnet
{
   struct NNetTrainingStatistics
   {
      // Percent of training runs that satisfied the training criteria
      double percent_successful;

      double mean_trainingset_perf;
      double std_trainingset_perf;

      double mean_validset_perf;
      double std_validset_perf;

      double mean_testset_perf;
      double std_testset_perf;

      std::vector<double> raw_trainingset_perf;
      std::vector<double> raw_validset_perf;
      std::vector<double> raw_testset_perf;
   };

   struct TrainedNNetRecord
   {
      std::string serialized_nnet;
      TrainingRecord training_record;
   };

   class TrainerUtils
   {
   public:
      static const unsigned int DEFAULT_TRAINING_RUNS = 1;
      static const unsigned int DEFAULT_SAVED_NNET_LIMIT = 1;

   private:

   public:
      TrainerUtils ();

      void set_training_runs (unsigned int _count);
      void set_saved_network_limit (unsigned int _count);
      const NNetTrainingStatistics &get_training_statistics (void) const;
      const vector<TrainedNNetRecord> &get_trained_neuralnets (void) const;

   protected:
      void collect_training_stats (const TrainingRecord &_tr);
      template<class _In, class _Out, template<class, class> class _NN>
      void save_network (_NN<_In, _Out> &_nnet, const TrainingRecord &_tr);


   protected:
      unsigned int &const_training_runs = training_runs;
      unsigned int &const_saved_nnet_limit = saved_nnet_limit;

   private:
      NNetTrainingStatistics training_stats;
      vector<TrainedNNetRecord> saved_networks;

      /**
       * The number of networks meeting the performance criteria to save. The best
       * performing trained networks will be saved up to the 'saved_nnet_limit' based on
       * the specified performance criteria.
       */
      unsigned int saved_nnet_limit;

      /**
       * The number of training runs to perform for each training set.
       */
      unsigned int training_runs;
   };

   template<class _In, class _Out, template<class, class> class _NN>
   void TrainerUtils::save_network (_NN<_In, _Out> &_nnet, const TrainingRecord &_tr)
   {
      std::cout << "BasicNNetTrainer::save_network() - entry\n";

      bool inserted = false;

      /*
       * Insert the neural network information into the list of best networks
       * is sorted order according to its performance.
       */
      for (auto it = begin (saved_networks); it != end (saved_networks); it++)
      {
         if (_tr.final_testset_performance () > it->training_record.final_testset_performance ())
         {
            TrainedNNetRecord c = {.serialized_nnet = _nnet.toJSON (), .training_record = _tr};
            saved_networks.emplace (it, c);
            inserted = true;
            saved_networks.resize (const_saved_nnet_limit);
            break;
         }
      }

      if (!inserted && saved_networks.size () < const_saved_nnet_limit)
         saved_networks.push_back ({.serialized_nnet = _nnet.toJSON (), .training_record = _tr});
   }

   inline
   const vector<TrainedNNetRecord> &TrainerUtils::get_trained_neuralnets (void) const
   {
      return saved_networks;
   }

}

#endif //FLEX_NEURALNET_TRAINERUTILS_H_
