//
// Created by kfedrick on 9/22/19.
//

#ifndef FLEX_NEURALNET_FATRAINER_H_
#define FLEX_NEURALNET_FATRAINER_H_

#include <cstddef>
#include <limits>
#include <memory>

#include "TrainerConfig.h"
#include "TrainerUtils.h"

#include "DataSet.h"
#include "Episode.h"
#include "NeuralNet.h"


namespace flexnnet
{
   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   class FATrainer : public TrainerUtils, public TrainerConfig, public _Eval<_NNIn, _NNOut, _Sample, _ErrFunc>
   {
   protected:
      using NN_Typ_ = NeuralNet<_NNIn, _NNOut>;
      using Exemplar_Typ_ = Exemplar<_NNIn, _NNOut>;
      using Episode_Typ_ = Episode<_NNIn, _NNOut>;
      using DataSet_Typ_ = DataSet<_NNIn, _NNOut, _Sample>;

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
       * @tparam _NN
       * @tparam _Exemplar
       * @param _nnet
       * @param _trnset
       */
      void train_epoch(_index_typ _epoch, NN_Typ_& _nnet, const DataSet_Typ_& _trnset);

      /**
       * Present one sample to the network from the training set and calculate
       * the appropriate weight adjustment.
       *
       * @param _nnet
       * @param _in
       * @param _out
       */
      void train_sample(_index_typ _epoch, NN_Typ_& _nnet, const Exemplar_Typ_& _sample);

      void train_sample(_index_typ _epoch, NN_Typ_& _nnet, const Episode_Typ_& _sample);

      void calc_weight_updates(const _NNOut& _tgt, const _NNOut _netout, NN_Typ_& _nnet);

      void update_weights(NN_Typ_& _nnet);

   private:
      std::shared_ptr<DataSet_Typ_> validation_dataset;
      std::shared_ptr<DataSet_Typ_> test_dataset;
   };

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::set_validation_dataset(const DataSet_Typ_& _vldset)
   {
      validation_dataset = &_vldset;
   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::set_test_dataset(const DataSet_Typ_& _tstset)
   {
      test_dataset = &_tstset;
   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::clear_validation_dataset(void)
   {

   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::clear_test_dataset(void)
   {

   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::train(NN_Typ_& _nnet, const DataSet_Typ_& _trnset)
   {
      std::cout << "FATrainer::train() - entry\n";
      TrainingRecord training_record;
      size_t no_runs = training_runs();

      for (size_t runndx = 0; runndx < no_runs; runndx++)
      {
         std::cout << "FATrainer.train() run " << runndx << "\n";
         training_record = train_run(_nnet, _trnset);

         // Save if one of best networks
         save_best_weights(training_record, _nnet);

         // TODO - update aggregate training statistics
      }
   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   TrainingRecord FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::train_run(NN_Typ_& _nnet, const DataSet_Typ_& _trnset)
   {
      std::cout << "   FATrainer::train_run() - entry\n";
      TrainingRecord training_record;

      double trn_perf, vld_perf, tst_perf;

      // Set reference to the performance value to use for best performance as:
      //    validation set performance if one is available; training set
      //    performance otherwise.
      //
      double& perf = (validation_dataset != nullptr)? vld_perf : trn_perf;

      // Previous performance value - used for failback testing
      double prev_trn_perf = std::numeric_limits<double>::max();
      double failback_limit = error_increase_limit();

      // Init best performance assuming we are trying to minimize error
      double best_perf { std::numeric_limits<double>::max() };
      size_t best_epoch = 0;

      // Iterate through training epochs
      _index_typ n_epochs = max_epochs();
      _index_typ epoch = 0;
      for (epoch = 0; epoch < n_epochs; epoch++)
      {
         std::cout << "   Enter - FATrainer::train_run() epoch " << epoch << "\n";

         // Save the network weights in case we need to fail back
         save_nnet_weights("failback", _nnet);

         // Call function to iterate over training samples and update
         // the network weights.
         train_epoch(epoch, _nnet, _trnset);

         // Evaluate the performance of the updated network
         trn_perf = _Eval<_NNIn, _NNOut, _Sample, _ErrFunc>::evaluate(_nnet, _trnset);

         if ((trn_perf - prev_trn_perf)/prev_trn_perf > failback_limit)
         {
            restore_nnet_weights("failback", _nnet);
            epoch--;
            continue;
         }

         training_record.training_set_trace.push_back({ .epoch=epoch, .performance=trn_perf });

         // Check the performance on the validation and test set if they exists
         if (validation_dataset != nullptr)
         {
            vld_perf = _Eval<_NNIn, _NNOut, _Sample, _ErrFunc>::evaluate(_nnet, *validation_dataset);
            training_record.training_set_trace.push_back({ .epoch=epoch, .performance=vld_perf });
         }

         if (test_dataset != nullptr)
         {
            tst_perf = _Eval<_NNIn, _NNOut, _Sample, _ErrFunc>::evaluate(_nnet, *test_dataset);
            training_record.training_set_trace.push_back({ .epoch=epoch, .performance=tst_perf });
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

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void
   FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::
      train_epoch(_index_typ _epoch, NN_Typ_& _nnet, const DataSet_Typ_& _trnset)
   {
      std::cout << "      Enter - FATrainer::train_epoch()\n";
      bool pending_updates = false;

      // Iterate through all samples in the training set_weights
      _index_typ sample_ndx = 0;
      for (auto exemplar : _trnset)
      {
         std::cout << "      FATrainer::train_run() - sample " << sample_ndx << "\n";
         train_sample(sample_ndx, _nnet, exemplar);
         pending_updates = true;

         // If training in online or mini-batch mode, update now.
         if (batch_mode() > 0 && sample_ndx % batch_mode() == 0)
         {
            update_weights(_nnet);
            pending_updates = false;
         }

         sample_ndx++;
      }

      // If training in batch mode or we there are updates pending from
      // an undersized mini-batch then update weights now
      if (batch_mode() == 0 || pending_updates)
         update_weights(_nnet);
   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::train_sample(_index_typ _epoch, NN_Typ_& _nnet, const Exemplar_Typ_& _exemplar)
   {
      std::cout << "         Enter - FATrainer::train_exemplar()\n";

      const _NNOut& nn_out = _nnet.activate(_exemplar.input());

      // TODO - nn_out and exemplar target must be 'vectorizable' - turn into
      // TODO - this will work for now if I use Datum
//      std::valarray<double>& gradient = _ErrFunc::gradient(nn_out(), _exemplar.target()());

      calc_weight_updates(_exemplar.target(), nn_out, _nnet);
   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::train_sample(_index_typ _epoch, NN_Typ_& _nnet, const Episode_Typ_& _exemplar)
   {
      std::cout << "         Enter - FATrainer::train_episode()\n";
   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::calc_weight_updates(const _NNOut& _tgt, const _NNOut _netout, NN_Typ_& _nnet)
   {
      std::cout << "            Enter - FATrainer::present_datum()\n";

   }

   template<class _NNIn, class _NNOut, template<class,class> class _Sample, class _ErrFunc, template<class,class,template<class,class> class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _Sample, _ErrFunc, _Eval>::update_weights(NN_Typ_& _nnet)
   {
      std::cout << "            Enter - FATrainer::update_weights()\n";
   }

}

#endif //FLEX_NEURALNET_FATRAINER_H_
