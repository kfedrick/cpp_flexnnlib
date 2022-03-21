//
// Created by kfedrick on 3/2/21.
//

#ifndef FLEX_NEURALNET_NEWSUPERVISEDTRAININGALGO_H_
#define FLEX_NEURALNET_NEWSUPERVISEDTRAININGALGO_H_

#include <memory>
#include <iostream>
#include <limits>
#include <flexnnet.h>
#include "DataSet.h"
#include "TrainingRecord.h"
#include "TrainingReport.h"
#include <BaseTrainer.h>
#include <TrainerConfig.h>
#include "BaseTrainer.h"
#include "Exemplar.h"
#include "ExemplarSeries.h"
#include "Reinforcement.h"

namespace flexnnet
{
   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   class BaseTrainingAlgo : public BaseTrainer, public TrainerConfig
   {
      using NNTyp = NeuralNet<InTyp, TgtTyp>;
      using DatasetTyp = DataSet<InTyp, TgtTyp, Sample>;
      using SampleTyp = Sample<InTyp, TgtTyp>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;

   public:
      BaseTrainingAlgo();

      LRPolicy& learning_rate_policy();
      const LRPolicy& learning_rate_policy() const;

      /**
       * Train the network the specified number of times using the
       * current weights initialization policy before each run. Save
       * the best set of trained networks. Calculate the training
       * statistics across all training runs.
       *
       * @param _trnset
       */
      void train(NNTyp& _nnet, const DatasetTyp& _trnset);

   protected:
      /**
       * Train the network once starting with weights initialized as
       * specified by the current policy.
       *
       * @param _trnset
       */
      TrainingRecord train_run(NNTyp& _nnet, const DatasetTyp& _trnset);

      /**
       * Train the network using all samples in the specified training
       * set.
       *
       * @param _epoch
       * @param _trnset
       */
      void train_epoch(size_t _epoch, NNTyp& _nnet, const DatasetTyp& _trnset);

      virtual void train_sample(NNTyp& _nnet, const SampleTyp& _sample) = 0;

      double update_performance_traces(unsigned int _epoch, double _trnperf, NNTyp& _nnet, TrainingRecord& _trec);

      void failback(NNTyp& _nnet);

   protected:
      virtual void alloc_working_memory(const NNTyp& _nnet) = 0;

   protected:
      LRPolicy learning_rate_policy_obj;
      FitFunc<InTyp, TgtTyp, Sample> fitnessfunc;

   private:
      DatasetTyp validation_dataset;
      DatasetTyp test_dataset;

      //std::map<std::string, Array2D<double>> weight_updates;

      TrainingRecord training_record;
   };

   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::BaseTrainingAlgo() : TrainerConfig() // :

      // LRPolicy(_nnet)//, nnet(_nnet)
   {
/*      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = nnet.get_layers();
      for (auto it: layers)
      {
         std::string id = it.first;

         // Set to train layer biases by default.
         TrainerConfig::set_train_biases(id, true);
      }*/
   }

   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   LRPolicy& BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::learning_rate_policy()
   {
      return learning_rate_policy_obj;
   }

   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   const LRPolicy& BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::learning_rate_policy() const
   {
      return learning_rate_policy_obj;
   }

   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   void BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::train(NNTyp& _nnet,
      const DatasetTyp& _trnset)
   {
      //std::cout << "SupervisedTrainingAlgo.train()\n" << std::flush;

      // TODO - validate neural network and data set are compatible.

      // Allocate training algo working memory for this neural network.
      this->alloc_working_memory(_nnet);

      // Initialize the learning rate policy working memory for this neural network.
      learning_rate_policy_obj.initialize(_nnet);

      // Initialize the train biases flag for this neural network.
      // TODO - this doesn't work right. The user cannot set the
      //    flags for individual layers prior to calling train OR after.
      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = _nnet.get_layers();
      for (auto it: layers)
      {
         std::string id = it.first;

         // Set to train layer biases by default.
         TrainerConfig::set_train_biases(id, true);
      }

      double trn_perf, trn_stdev;
      double perf;

      size_t no_runs = TrainerConfig::training_runs();

      // If no randomization of training order is set then
      // make certain training set order is normalized now.
      if (!this->randomize_training_order())
         _trnset.normalize_order();

      for (size_t runndx = 0; runndx < no_runs; runndx++)
      {
         training_record.clear();

         if (runndx > 0)
            _nnet.initialize_weights();

         save_network_weights(_nnet, "initial_weights");

         // *** train the network
         train_run(_nnet, _trnset);

         const std::map<std::string, std::shared_ptr<NetworkLayer>>
            & layers = _nnet.get_layers();
         for (auto& it: layers)
            training_record.network_weights[it.first] = it.second->weights();

         save_training_record(training_record);

         // TODO - update aggregate training statistics
      }
      //std::cout << "SupervisedTrainingAlgo.train() EXIT\n" << std::flush;
   }

   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   TrainingRecord BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::train_run(NNTyp& _nnet,
      const DatasetTyp& _trnset)
   {
      //std::cout << "BaseTrainingAlgo.train_one_run() ENTRY\n" << std::flush;

      double trn_perf, trn_stdev;
      double trn_perf_improvement;
      double perf;

      unsigned int failback_count = 0;

      learning_rate_policy_obj.clear_learning_rate_adjustments();
      learning_rate_policy_obj.init_learning_rate();

      // Previous performance vectorize - used for failback testing
      double prev_trn_perf = std::numeric_limits<double>::max();
      double failback_limit = TrainerConfig::error_increase_limit();

      // Init best performance assuming we are trying to minimize error
      training_record.stop_signal = TrainingStopSignal::UNKNOWN;

      /*
       * Evaluate and save the performance for the initial network
       */
      trn_perf = fitnessfunc.calc_fitness(_nnet, _trnset);
      perf = update_performance_traces(0, trn_perf, _nnet, training_record);

      training_record.best_epoch = 0;
      training_record.best_performance = perf;
      save_network_weights(_nnet, "best_epoch");

      // Iterate through training epochs
      size_t n_epochs = TrainerConfig::max_epochs();
      size_t epoch = 0;
      for (epoch = 0; epoch < n_epochs; epoch++)
      {
         //std::cout << "BaseTrainingAlgo : epoch  " << epoch << "\n" << std::flush;

         // Save the network weights in case we need to fail back
         save_network_weights(_nnet, "failback");

         // Call function to iterate over training samples and update
         // the network weights.
         train_epoch(epoch, _nnet, _trnset);

         // Evaluate the performance of the updated network
         trn_perf = fitnessfunc.calc_fitness(_nnet, _trnset);

         /*
          * If the performance on the training set worsens by an
          * amount greater than the fail-back limit then (1) restore
          * the previous weights, (2) lower the learning rates and
          * retry the epoch.
          */
         trn_perf_improvement = (prev_trn_perf > 0) ? (trn_perf - prev_trn_perf) / prev_trn_perf :
                                (trn_perf - 1e-9) / 1e-9;
         if (trn_perf_improvement > failback_limit)
         {
            failback(_nnet);
            epoch--;

            failback_count++;
            if (failback_count > this->max_failbacks())
            {
               training_record.stop_signal = TrainingStopSignal::MAX_FAILBACK_REACHED;
               break;
            }
            continue;
         }
         else
         {
            failback_count = 0;
            learning_rate_policy_obj.apply_learning_rate_adjustments();
         }

         // Update performance history in training record
         if (epoch < 10 || epoch % TrainerConfig::report_frequency()
                           == 0 || epoch == n_epochs - 1)
            perf = update_performance_traces(epoch + 1, trn_perf, _nnet, training_record);

         // Call function to save network weights for the best epoch.
         if (perf < training_record.best_performance)
         {
            training_record.best_epoch = epoch + 1;
            training_record.best_performance = perf;
            if (perf < TrainerConfig::error_goal())
               training_record.stop_signal = TrainingStopSignal::CRITERIA_MET;

            save_network_weights(_nnet, "best_epoch");
         }

         // If we've reached the target error goal then exit.
         /*
         if (perf < TrainerConfig::error_goal())
         {
            training_record.stop_signal = TrainingStopSignal::CRITERIA_MET;
            break;
         }*/
      }

      if (training_record.stop_signal == TrainingStopSignal::UNKNOWN)
         training_record.stop_signal = TrainingStopSignal::MAX_EPOCHS_REACHED;

      // Restore the best network weights.
      //restore_network_weights(nnet, "best_epoch");

      //std::cout << "SupervisedTrainingAlgo.train_one_run() EXIT\n" << std::flush;
      return training_record;
   }

   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   void BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::train_epoch(
      size_t _epoch, NNTyp& _nnet, const DatasetTyp& _trnset)
   {
      //std::cout << "BaseTrainingAlgo.train_epoch(Exemplar)\n" << std::flush;

      bool pending_updates = false;

      if (this->randomize_training_order())
         _trnset.randomize_order();

      // Iterate through all samples in the training set_weights
      size_t sample_ndx = 0;
      for (auto& sample: _trnset)
      {
         train_sample(_nnet, sample);
         pending_updates = true;

         // If training in online or mini-batch mode, update now.
         if (TrainerConfig::batch_mode() > 0 && sample_ndx % TrainerConfig::batch_mode() == 0)
         {
            adjust_network_weights(_nnet);
            pending_updates = false;
         }

         sample_ndx++;
      }

      // If training in batch mode or we there are updates pending from
      // an undersized mini-batch then update weights now
      if (TrainerConfig::batch_mode() == 0 || pending_updates)
         adjust_network_weights(_nnet);
   }




/*

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class Sample,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class, template<class> class>
      class Eval,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   SupervisedTrainingAlgo<InTyp,
                          TgtTyp,
                          Sample,
                          NN,
                          Dataset,
                          Eval,
                          FitFunc,
                          LRPolicy>::train_exemplar(const Exemplar<InTyp,TgtTyp>& _exemplar)
   {
      std::cout << "SupervisedTrainerAlgo.train_exemplar()\n" << std::flush;

      const NNFeatureSet<TgtTyp>& nn_out = this->nnet.activate(_exemplar.first);

      //const std::map<std::string, std::valarray<double>>& nnoutv_map = nn_out.value_map();
      const std::map<std::string, std::valarray<double>>
         & targetv_map = _exemplar.second.value_map();

      ValarrMap gradient;
      evaluator.calc_error_gradient(_exemplar.second, _exemplar.second, nn_out, gradient);
      this->calc_weight_updates(gradient);
      LRPolicy::calc_learning_rate_adjustment(0);
   }
   */

/*
   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class Sample,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class, template<class> class>
      class Eval,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   SupervisedTrainingAlgo<InTyp,
                          TgtTyp,
                          Sample,
                          NN,
                          Dataset,
                          Eval,
                          FitFunc,
                          LRPolicy>::
   calc_weight_updates(const BaseNeuralNet& _nnet, const ValarrMap _egradient)
   {
      this->nnet.backprop(_egradient);

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & layers = this->nnet.get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;
         Array2D<double> lr = LRPolicy::get_learning_rates(id);

         const Array2D<double> dE_dw = it.second->dEdw();

         const Array2D<double>::Dimensions dims = weight_updates[id].size();

         // If this layer doesn't train biases, stop before the last column
         unsigned int
            last_col = (TrainerConfig::train_biases(id)) ? dims.cols : dims.cols - 1;

         weight_updates[id] = 0;
         for (unsigned int row = 0; row < dims.rows; row++)
            for (unsigned int col = 0; col < last_col; col++)
               weight_updates[id].at(row, col) = -lr.at(row, col) * dE_dw.at(row, col);

         accumulate_weight_updates(nnet, id, weight_updates[id]);
      }
   }
*/

/*
   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   void BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::alloc_working_memory()
   {
/*      weight_updates.clear();

      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = this->nnet.get_layers();
      for (auto it: layers)
      {
         std::string id = it.first;
         const LayerWeights& w = it.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         weight_updates[id] = {};
         weight_updates[id].resize(dim.rows, dim.cols);
      }
   }
   */

   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   double BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::update_performance_traces(
      unsigned int _epoch, double _trnperf, NNTyp& _nnet, TrainingRecord& _trec)
   {
      double vld_perf, tst_perf;
      double vld_stdev, tst_stdev;

      double perf = _trnperf;

      // Save the training set performance for this epoch
      _trec.training_set_trace.push_back({.epoch = _epoch, .performance = _trnperf});

      // Record the performance on the validation set for this epoch
      if (validation_dataset.size() > 0)
      {
         //std::tie(vld_perf, vld_stdev) = fitnessfunc.calc_fitness(this->nnet, validation_dataset);
         vld_perf = fitnessfunc.calc_fitness(_nnet, validation_dataset);

         // If validation set exist use it as the overall performance measure
         // in order to determine best weights using early stopping.
         perf = vld_perf;

         _trec.validation_set_trace.push_back({.epoch = _epoch, .performance = vld_perf});
      }

      // Record the performance on the test set for this epoch
      if (test_dataset.size() > 0)
      {
         //std::tie(tst_perf, tst_stdev) = evaluator.evaluate(this->nnet, test_dataset);
         tst_perf = fitnessfunc.calc_fitness(_nnet, test_dataset);
         _trec.test_set_trace.push_back({.epoch = _epoch, .performance = tst_perf});
      }

      return perf;
   }

   template<class InTyp, class TgtTyp, template<class, class> class Sample,
      template<class, class, template<class, class> class> class FitFunc, class LRPolicy>
   void BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::failback(NNTyp& _nnet)
   {
      std::cout << "fail-back!!!!\n";
      restore_network_weights(_nnet, "failback");
      learning_rate_policy_obj.reduce_learning_rate();
   }

} // end namespace flexnnet

#endif //FLEX_NEURALNET_SUPERVISEDTRAININGALGO_H_