//
// Created by kfedrick on 10/8/19.
//

#ifndef FLEX_NEURALNET_BASICTRAINER_H_
#define FLEX_NEURALNET_BASICTRAINER_H_

#include <network/include/NeuralNet.h>
#include <datasets/include/DataSet2.h>
#include <iostream>
#include "TrainingRecord.h"
#include "TrainerConfig.h"

namespace flexnnet
{
   template<class _NNIn, class _NNOut, template<class, class> class _TData,
      template<class, class, template<class, class> class, template<class> class> class _TrainAlgo,
      template<class, class, template<class, class> class, template<class> class> class _Eval,
      template<class> class _ErrFunc = RMSError>
   class BasicTrainer
      : public TrainerConfig,
        public TrainerUtils,
        public _TrainAlgo<_NNIn, _NNOut, _TData, _ErrFunc>,
        public _Eval<_NNIn, _NNOut, _TData, _ErrFunc>
   {
   protected:
      using NN_Typ_ = NeuralNet<_NNIn, _NNOut>;
      using DataSet_Typ_ = DataSet<_NNIn, _NNOut, _TData>;
      using TrainAlgo_Typ_ = _TrainAlgo<_NNIn, _NNOut, _TData, _ErrFunc>;

   public:
      /**
       * Set a validation data set to be used during training. If the validation
       * data set is used then the best network performance will be determined
       * based on the performance on the validation data set.
       *
       * @param _vldset
       */
      void set_validation_dataset(const DataSet_Typ_& _vldset);

      /**
       * Set a test data set to be used during training. If a test data set is
       * assigned then the network will evaluate and report the performance of
       * the test set during training.
       *
       * @param _tstset
       */
      void set_test_dataset(const DataSet_Typ_& _tstset);

      /**
       * Remove the validation dataset.
       */
      void clear_validation_dataset(void);

      /**
       * Remove the test dataset from the trainer.
       */
      void clear_test_dataset(void);

      /**
       * Perform the number of training runs on the neural network specified by
       * 'training_runs()' using the specified training set. Save the network
       * weights from the best performing runs up to 'max_saved_nnet_limit'.
       *
       * @param _nnet
       * @param _trnset
       */
      void train(NN_Typ_& _nnet, const DataSet_Typ_& _trnset);

   protected:
      /**
       * Train neural network for up to the maximum specified epochs or until the
       * convergence criteria is met.
       *
       * @param _nnet
       * @param _trnset
       * @return
       */
      TrainingRecord train_run(NN_Typ_& _nnet, const DataSet_Typ_& _trnset);

      /**
       * Perform one training epoch by presenting each sample in the training set
       * to the network and updating the weights.
       *
       * @tparam _NNIn
       * @tparam _NNOut
       * @tparam _Exemplar
       * @param _nnet
       * @param _trnset
       */
      void train_epoch(size_t _epoch, NN_Typ_& _nnet, const DataSet_Typ_& _trnset);

   private:
      std::shared_ptr<DataSet_Typ_> validation_dataset;
      std::shared_ptr<DataSet_Typ_> test_dataset;
   };

   template<class _NNIn, class _NNOut, template<class, class> class _TData,
      template<class, class, template<class, class> class, template<class> class> class _TrainAlgo,
      template<class, class, template<class, class> class, template<class> class> class _Eval,
      template<class> class _ErrFunc>
   void
   BasicTrainer<_NNIn, _NNOut, _TData, _TrainAlgo, _Eval, _ErrFunc>::set_validation_dataset(const DataSet_Typ_& _vldset)
   {
      validation_dataset = &_vldset;
   }

   template<class _NNIn, class _NNOut, template<class, class> class _TData,
      template<class, class, template<class, class> class, template<class> class> class _TrainAlgo,
      template<class, class, template<class, class> class, template<class> class> class _Eval,
      template<class> class _ErrFunc>
   void BasicTrainer<_NNIn, _NNOut, _TData, _TrainAlgo, _Eval, _ErrFunc>::set_test_dataset(const DataSet_Typ_& _tstset)
   {
      test_dataset = &_tstset;
   }

   template<class _NNIn, class _NNOut, template<class, class> class _TData,
      template<class, class, template<class, class> class, template<class> class> class _TrainAlgo,
      template<class, class, template<class, class> class, template<class> class> class _Eval,
      template<class> class _ErrFunc>
   void BasicTrainer<_NNIn, _NNOut, _TData, _TrainAlgo, _Eval, _ErrFunc>::clear_validation_dataset(void)
   {

   }

   template<class _NNIn, class _NNOut, template<class, class> class _TData,
      template<class, class, template<class, class> class, template<class> class> class _TrainAlgo,
      template<class, class, template<class, class> class, template<class> class> class _Eval,
      template<class> class _ErrFunc>
   void BasicTrainer<_NNIn, _NNOut, _TData, _TrainAlgo, _Eval, _ErrFunc>::clear_test_dataset(void)
   {

   }

   template<class _NNIn, class _NNOut, template<class, class> class _TData,
      template<class, class, template<class, class> class, template<class> class> class _TrainAlgo,
      template<class, class, template<class, class> class, template<class> class> class _Eval,
      template<class> class _ErrFunc>
   void
   BasicTrainer<_NNIn, _NNOut, _TData, _TrainAlgo, _Eval, _ErrFunc>::train(NN_Typ_& _nnet, const DataSet_Typ_& _trnset)
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

   template<class _NNIn, class _NNOut, template<class, class> class _TData,
      template<class, class, template<class, class> class, template<class> class> class _TrainAlgo,
      template<class, class, template<class, class> class, template<class> class> class _Eval,
      template<class> class _ErrFunc>
   TrainingRecord
   BasicTrainer<_NNIn,
                _NNOut,
                _TData,
                _TrainAlgo,
                _Eval,
                _ErrFunc>::train_run(NN_Typ_& _nnet, const DataSet_Typ_& _trnset)
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
         trn_perf = _Eval<_NNIn, _NNOut, _TData, _ErrFunc>::evaluate(_nnet, _trnset);

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
            vld_perf = _Eval<_NNIn, _NNOut, _TData, _ErrFunc>::evaluate(_nnet, *validation_dataset);
            training_record.training_set_trace.push_back({.epoch=epoch, .performance=vld_perf});
         }

         if (test_dataset != nullptr)
         {
            tst_perf = _Eval<_NNIn, _NNOut, _TData, _ErrFunc>::evaluate(_nnet, *test_dataset);
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

   template<class _NNIn, class _NNOut, template<class, class> class _TData,
      template<class, class, template<class, class> class, template<class> class> class _TrainAlgo,
      template<class, class, template<class, class> class, template<class> class> class _Eval,
      template<class> class _ErrFunc>
   void
   BasicTrainer<_NNIn,
                _NNOut,
                _TData,
                _TrainAlgo,
                _Eval,
                _ErrFunc>::train_epoch(size_t _epoch, NN_Typ_& _nnet, const DataSet_Typ_& _trnset)
   {
      std::cout << "      Enter - BasicTrainer::train_epoch()\n";
      bool pending_updates = false;

      // Iterate through all samples in the training set_weights
      size_t sample_ndx = 0;
      for (auto exemplar : _trnset)
      {
         std::cout << "      BasicTrainer::train_epoch() - sample " << sample_ndx << "\n";
         TrainAlgo_Typ_::present_datum(sample_ndx, _nnet, exemplar);
         pending_updates = true;

         // If training in online or mini-batch mode, update now.
         if (batch_mode() > 0 && sample_ndx % batch_mode() == 0)
         {
            TrainAlgo_Typ_::update_weights(_nnet);
            pending_updates = false;
         }

         sample_ndx++;
      }

      // If training in batch mode or we there are updates pending from
      // an undersized mini-batch then update weights now
      if (batch_mode() == 0 || pending_updates)
         TrainAlgo_Typ_::update_weights(_nnet);
   }
}

#endif //FLEX_NEURALNET_BASICTRAINER_H_
