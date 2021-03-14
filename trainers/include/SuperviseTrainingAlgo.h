//
// Created by kfedrick on 3/2/21.
//

#ifndef FLEX_NEURALNET_SUPERVISEDTRAININGALGO_H_
#define FLEX_NEURALNET_SUPERVISEDTRAININGALGO_H_

#include <memory>
#include <iostream>
#include <limits>
#include <flexnnet.h>
#include "EnumeratedDataSet.h"
#include "TrainingRecord.h"
#include "TrainerConfig.h"

namespace flexnnet
{
   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
         template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc>

   class SuperviseTrainingAlgo : public TrainerConfig
   {
      using _NNTyp = _NN<_InTyp,_OutTyp>;
      using _DatasetTyp = _DataSet<_InTyp,_OutTyp>;
      using _Exemplar = std::tuple<_InTyp,_OutTyp>;

   public:
      /**
       * Train the network the specified number of times using the
       * current weights initialization policy before each run. Save
       * the best set of trained networks. Calculate the training
       * statistics across all training runs.
       *
       * @param _nnet
       * @param _trnset
       */
      void train(_NNTyp& _nnet, const _DatasetTyp& _trnset);

   protected:
      /**
       * Train the network once starting with weights initialized as
       * specified by the current policy.
       *
       * @param _nnet
       * @param _trnset
       */
      TrainingRecord train_run(_NNTyp& _nnet, const _DatasetTyp& _trnset);

      /**
       * Train the network using all samples in the specified training
       * set.
       *
       * @param _epoch
       * @param _nnet
       * @param _trnset
       */
      void train_epoch(size_t _epoch, _NNTyp& _nnet, const _DatasetTyp& _trnset);

      /**
       * Present a single sample from the training set to the network and
       * calculate the network weight updates.
       *
       * @param _epoch
       * @param _nnet
       * @param _sample
       */
      void train_sample(_NNTyp& _nnet, const _Exemplar& _sample);

   private:
      _DatasetTyp validation_dataset;
      _DatasetTyp test_dataset;
   };

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc>
   void SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc>::train(_NNTyp& _nnet, const _DatasetTyp& _trnset)
   {
      std::cout << "BasicTrainer::train() - entry\n";
      TrainingRecord training_record;
      size_t no_runs = training_runs();

      for (size_t runndx = 0; runndx < no_runs; runndx++)
      {
         std::cout << "BasicTrainer.train() run " << runndx << "\n";
         training_record = train_run(_nnet, _trnset);

         // Save if one of best networks
         save_best_weights(training_record, _nnet);

         // TODO - update aggregate training statistics
      }
   }

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc>
   TrainingRecord SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc>::train_run(_NNTyp& _nnet, const _DatasetTyp& _trnset)
   {
      std::cout << "   BasicTrainer::train_run() - entry\n";
      TrainingRecord training_record;

      double trn_perf, vld_perf, tst_perf;

      // Set reference to the performance value to use for best performance as:
      //    validation set performance if one is available; training set
      //    performance otherwise.
      //
      double& perf = (validation_dataset != nullptr) ? vld_perf : trn_perf;

      // Previous performance value - used for failback testing
      double prev_trn_perf = std::numeric_limits<double>::max();
      double failback_limit = error_increase_limit();

      // Init best performance assuming we are trying to minimize error
      double best_perf{std::numeric_limits<double>::max()};
      size_t best_epoch = 0;

      // Iterate through training epochs
      size_t n_epochs = max_epochs();
      size_t epoch = 0;
      for (epoch = 0; epoch < n_epochs; epoch++)
      {
         std::cout << "   Enter - FATrainer::train_run() epoch " << epoch << "\n";

         // Save the network weights in case we need to fail back
         save_nnet_weights("failback", _nnet);

         // Call function to iterate over training samples and update
         // the network weights.
         train_epoch(epoch, _nnet, _trnset);

         // Evaluate the performance of the updated network
         //trn_perf = _Eval<_NNIn, _NNOut, _TData, _ErrFunc>::evaluate(_nnet, _trnset);

         if ((trn_perf - prev_trn_perf) / prev_trn_perf > failback_limit)
         {
            restore_nnet_weights("failback", _nnet);
            epoch--;
            continue;
         }

         training_record.training_set_trace.push_back({.epoch=epoch, .performance=trn_perf});

         // Check the performance on the validation and test set if they exists
         if (validation_dataset != nullptr)
         {
            //vld_perf = _Eval<_NNIn, _NNOut, _TData, _ErrFunc>::evaluate(_nnet, *validation_dataset);
            training_record.training_set_trace.push_back({.epoch=epoch, .performance=vld_perf});
         }

         if (test_dataset != nullptr)
         {
            //tst_perf = _Eval<_NNIn, _NNOut, _TData, _ErrFunc>::evaluate(_nnet, *test_dataset);
            training_record.training_set_trace.push_back({.epoch=epoch, .performance=tst_perf});
         }

         // Call function to save network weights for the best epoch.
         if (perf < training_record.best_performance)
         {
            training_record.best_epoch = epoch;
            training_record.best_performance = perf;
            save_nnet_weights("best_epoch", _nnet);
         }
      }

      // Restore the best network weights.
      restore_nnet_weights("best_epoch", _nnet);

      return training_record;
   }

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc>
   void SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc>::train_epoch(size_t _epoch, _NNTyp& _nnet, const _DatasetTyp& _trnset)
   {
      std::cout << "      Enter - BasicTrainer::train_epoch()\n";
      bool pending_updates = false;

      // Iterate through all samples in the training set_weights
      size_t sample_ndx = 0;
      for (auto exemplar : _trnset)
      {
         std::cout << "      BasicTrainer::train_epoch() - sample " << sample_ndx << "\n";
         train_sample(sample_ndx, _nnet, exemplar);
         pending_updates = true;

         // If training in online or mini-batch mode, update now.
         if (batch_mode() > 0 && sample_ndx % batch_mode() == 0)
         {
            //TrainAlgo_Typ_::update_weights(_nnet);
            pending_updates = false;
         }

         sample_ndx++;
      }

      // If training in batch mode or we there are updates pending from
      // an undersized mini-batch then update weights now
      //if (batch_mode() == 0 || pending_updates)
      //   TrainAlgo_Typ_::update_weights(_nnet);
   }

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc>
   void SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc>::train_sample(_NNTyp& _nnet, const _Exemplar& _exemplar)
   {
      const _OutTyp& nn_out = _nnet.activate(_exemplar.first);
      const std::valarray<double>& nnoutv_map = nn_out.vectorize();
      const std::valarray<double>& targetv_map = _exemplar.second.vectorize();
      ValarrMap gradient = _FitnessFunc<_OutTyp>::calc_error_gradient(targetv_map, nnoutv_map);
   }
}
#endif //FLEX_NEURALNET_SUPERVISEDTRAININGALGO_H_
