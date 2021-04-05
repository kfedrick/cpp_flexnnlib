//
// Created by kfedrick on 3/2/21.
//

#ifndef FLEX_NEURALNET_SUPERVISEDTRAININGALGO_H_
#define FLEX_NEURALNET_SUPERVISEDTRAININGALGO_H_

#include <memory>
#include <iostream>
#include <limits>
#include <flexnnet.h>
#include "DataSet.h"
#include "TrainingRecord.h"
#include "TrainingReport.h"
#include "BaseTraining.h"

namespace flexnnet
{
   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
         template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc,
      class _LRPolicy>

   class SuperviseTrainingAlgo : public BaseTraining, public _LRPolicy
   {
      using _NNTyp = _NN<_InTyp,_OutTyp>;
      using _DatasetTyp = _DataSet<_InTyp,_OutTyp>;
      using _Exemplar = std::pair<_InTyp,_OutTyp>;

   public:
      SuperviseTrainingAlgo(_NNTyp& _nnet) : _LRPolicy(_nnet), nnet(_nnet)
      {
         std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = nnet.get_layers();
         for (auto it : layers)
         {
            std::string id = it.first;

            // Set to train layer biases by default.
            set_train_biases(id, true);
         }
      }

      /**
       * Train the network the specified number of times using the
       * current weights initialization policy before each run. Save
       * the best set of trained networks. Calculate the training
       * statistics across all training runs.
       *
       * @param _nnet
       * @param _trnset
       */
      void train(const _DatasetTyp& _trnset);

   protected:
      /**
       * Train the network once starting with weights initialized as
       * specified by the current policy.
       *
       * @param _nnet
       * @param _trnset
       */
      TrainingRecord train_run(const _DatasetTyp& _trnset);

      /**
       * Train the network using all samples in the specified training
       * set.
       *
       * @param _epoch
       * @param _nnet
       * @param _trnset
       */
      void train_epoch(size_t _epoch, const _DatasetTyp& _trnset);

      /**
       * Present a single sample from the training set to the network and
       * calculate the network weight updates.
       *
       * @param _epoch
       * @param _nnet
       * @param _sample
       */
      void train_sample(const _Exemplar& _sample);
      
      virtual void calc_weight_updates(const ValarrMap _egradient, BaseNeuralNet& _nnet) override;

   private:
      void initialize();

   private:
      _NNTyp& nnet;

      _Eval<_InTyp, _OutTyp, _NN, _DataSet, _FitnessFunc> evaluator;

      _DatasetTyp validation_dataset;
      _DatasetTyp test_dataset;

      std::map<std::string, Array2D<double>> weight_updates;

      TrainingRecord training_record;
   };


   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc,
      class _LRPolicy>
   void SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc, _LRPolicy>::train(const _DatasetTyp& _trnset)
   {
      double trn_perf, vld_perf, tst_perf;
      double trn_stdev, vld_stdev, tst_stdev;

      initialize();

      size_t no_runs = training_runs();

      save_nnet_weights("initial_weights", nnet);

      for (size_t runndx = 0; runndx < no_runs; runndx++)
      {
         training_record.clear();

         if (runndx > 0)
            nnet.initialize_weights();

         /*
          * Evaluate the initial performance
          */
         double& perf = (validation_dataset.size() > 0) ? vld_perf : trn_perf;

         std::tie(trn_perf,trn_stdev) = evaluator.evaluate(nnet, _trnset);
         training_record.training_set_trace.push_back({.epoch=0, .performance=trn_perf});
         // Record the performance on the validation set for this epoch

         if (validation_dataset.size() > 0)
         {
            std::tie(vld_perf,vld_stdev) = evaluator.evaluate(nnet, validation_dataset);
            training_record.validation_set_trace.push_back({.epoch=0, .performance=vld_perf});
         }

         // Record the performance on the test set for this epoch
         if (test_dataset.size() > 0)
         {
            std::tie(tst_perf,tst_stdev) = evaluator.evaluate(nnet, test_dataset);
            training_record.test_set_trace.push_back({.epoch=0, .performance=tst_perf});
         }

         training_record.best_epoch = 0;
         training_record.best_performance = trn_perf;
         save_nnet_weights("best_epoch", nnet);

         train_run(_trnset);

         // Save if one of best networks
         training_record.network_weights = nnet.get_weights();
         save_best_weights(training_record, nnet);

         // TODO - update aggregate training statistics
      }
   }

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc,
      class _LRPolicy>
   TrainingRecord SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc, _LRPolicy>::train_run(const _DatasetTyp& _trnset)
   {
      double trn_perf, vld_perf, tst_perf;
      double trn_stdev, vld_stdev, tst_stdev;
      double trn_perf_improvement;

      _LRPolicy::reset();

      // Set reference to the performance value to use for best performance as:
      //    validation set performance if one is available; training set
      //    performance otherwise.
      //
      double& perf = (validation_dataset.size() > 0) ? vld_perf : trn_perf;

      // Previous performance value - used for failback testing
      double prev_trn_perf = std::numeric_limits<double>::max();
      double failback_limit = error_increase_limit();

      // Init best performance assuming we are trying to minimize error
      //training_record.best_performance = std::numeric_limits<double>::max();
      training_record.stop_signal = TrainingStopSignal::UNKNOWN;

      // Iterate through training epochs
      size_t n_epochs = max_epochs();
      size_t epoch = 0;
      for (epoch = 0; epoch < n_epochs; epoch++)
      {
         // Save the network weights in case we need to fail back
         save_nnet_weights("failback", nnet);

         // Call function to iterate over training samples and update
         // the network weights.
         train_epoch(epoch, _trnset);

         // Evaluate the performance of the updated network
         std::tie(trn_perf,trn_stdev) = evaluator.evaluate(nnet, _trnset);

         /*
          * If the performance on the training set worsens by an
          * amount greater than the fail-back limit then (1) restore
          * the previous weights, (2) lower the learning rates and
          * retry the epoch.
          */
         trn_perf_improvement = (prev_trn_perf > 0) ? (trn_perf - prev_trn_perf) / prev_trn_perf : (trn_perf - 1e-9) / 1e-9;
         if (trn_perf_improvement > failback_limit)
         {
            restore_nnet_weights("failback", nnet);
            _LRPolicy::reduce_learning_rate();
            epoch--;
            continue;
         }

         // Record the performance on the training set for this epoch
         if (epoch < 10 || epoch % report_frequency() == 0 || epoch == n_epochs-1)
            training_record.training_set_trace.push_back({.epoch=epoch+1, .performance=trn_perf});

         // Record the performance on the validation set for this epoch
         if (validation_dataset.size() > 0)
         {
            std::tie(vld_perf,vld_stdev) = evaluator.evaluate(nnet, validation_dataset);
            if (epoch < 10 || epoch % report_frequency() == 0 || epoch == n_epochs-1)
               training_record.validation_set_trace.push_back({.epoch=epoch+1, .performance=vld_perf});
         }

         // Record the performance on the test set for this epoch
         if (test_dataset.size() > 0)
         {
            std::tie(tst_perf,tst_stdev) = evaluator.evaluate(nnet, test_dataset);
            if (epoch < 10 || epoch % report_frequency() == 0 || epoch == n_epochs-1)
               training_record.test_set_trace.push_back({.epoch=epoch+1, .performance=tst_perf});
         }

         // Call function to save network weights for the best epoch.
         if (perf < training_record.best_performance)
         {
            training_record.best_epoch = epoch+1;
            training_record.best_performance = perf;
            if (perf < error_goal())
               training_record.stop_signal = TrainingStopSignal::CRITERIA_MET;

            save_nnet_weights("best_epoch", nnet);
         }

         // If we've reached the target error goal then exit.
         if (perf < error_goal())
         {
            training_record.stop_signal = TrainingStopSignal::CRITERIA_MET;
            break;
         }

         // Apply adjustments to the learning rates
         _LRPolicy::apply_learning_rate_adjustments();
      }

      if (training_record.stop_signal == TrainingStopSignal::UNKNOWN)
         training_record.stop_signal = TrainingStopSignal::MAX_EPOCHS_REACHED;

      // Restore the best network weights.
      restore_nnet_weights("best_epoch", nnet);

      return training_record;
   }

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc,
      class _LRPolicy>
   void SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc, _LRPolicy>::train_epoch(size_t _epoch, const _DatasetTyp& _trnset)
   {
      bool pending_updates = false;

      // Iterate through all samples in the training set_weights
      size_t sample_ndx = 0;
      for (auto& exemplar : _trnset)
      {
         train_sample(exemplar);
         pending_updates = true;

         // If training in online or mini-batch mode, update now.
         if (batch_mode() > 0 && sample_ndx % batch_mode() == 0)
         {
            adjust_network_weights(nnet);
            pending_updates = false;
         }

         sample_ndx++;
      }

      // If training in batch mode or we there are updates pending from
      // an undersized mini-batch then update weights now
      if (batch_mode() == 0 || pending_updates)
         adjust_network_weights(nnet);
   }

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc,
      class _LRPolicy>
   void SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc, _LRPolicy>::train_sample(const _Exemplar& _exemplar)
   {
      const _OutTyp& nn_out = nnet.activate(_exemplar.first);

      const std::map<std::string, std::valarray<double>>& nnoutv_map = nn_out.value_map();
      const std::map<std::string, std::valarray<double>>& targetv_map = _exemplar.second.value_map();

      ValarrMap gradient = evaluator.calc_error_gradient(targetv_map, nnoutv_map);

      calc_weight_updates(gradient, nnet);
      _LRPolicy::calc_learning_rate_adjustment(0);
   }

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc,
      class _LRPolicy>
   void SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc, _LRPolicy>::
   calc_weight_updates(const ValarrMap _egradient, BaseNeuralNet& _nnet)
   {
      _nnet.backprop(_egradient);

      std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = _nnet.get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;
         Array2D<double> lr = _LRPolicy::get_learning_rates(id);

         const LayerState& state = it.second->layer_state();
         const std::valarray<double>& netin_errorv = state.netin_errorv;
         const Array2D<double> dE_dw = it.second->dE_dw();

         const Array2D<double>::Dimensions dims = weight_updates[id].size();

         // If this layer doesn't train biases, stop before the last column
         unsigned int last_col = (train_biases(id)) ? dims.cols : dims.cols-1;

         weight_updates[id] = 0;
         for (unsigned int row = 0; row < dims.rows; row++)
            for (unsigned int col = 0; col < last_col; col++)
               weight_updates[id].at(row, col) = -lr.at(row,col) * dE_dw.at(row,col);
      }
      accumulate_weight_updates(weight_updates);

   }

   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class, class, template<class,class> class,
      template<class,class> class, template<class> class> class _Eval,
      template<class> class _FitnessFunc,
      class _LRPolicy>
   void SuperviseTrainingAlgo<_InTyp, _OutTyp, _NN, _DataSet, _Eval, _FitnessFunc,_LRPolicy>::initialize()
   {
      BaseTraining::initialize(nnet);

      weight_updates.clear();

      std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = nnet.get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;
         LayerWeights& w = it.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         weight_updates[id] = {};
         weight_updates[id].resize(dim.rows, dim.cols);
      }
   }
}
#endif //FLEX_NEURALNET_SUPERVISEDTRAININGALGO_H_
