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
   class SupervisedTrainingAlgo : public BaseTrainer, public TrainerConfig, public LRPolicy
   {
      using DatasetTyp = Dataset<InTyp, TgtTyp, Sample>;
      using SampleTyp = Sample<InTyp, TgtTyp>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;

   public:
      SupervisedTrainingAlgo(NN<InTyp, TgtTyp>& _nnet);

      /**
       * Train the network the specified number of times using the
       * current weights initialization policy before each run. Save
       * the best set of trained networks. Calculate the training
       * statistics across all training runs.
       *
       * @param _trnset
       */
      void
      train(const DatasetTyp& _trnset);

   protected:
      /**
       * Train the network once starting with weights initialized as
       * specified by the current policy.
       *
       * @param _trnset
       */
      TrainingRecord
      train_run(const DatasetTyp& _trnset);

      /**
       * Train the network using all samples in the specified training
       * set.
       *
       * @param _epoch
       * @param _trnset
       */
      void
      train_epoch(size_t _epoch, const Dataset<InTyp,TgtTyp,Exemplar>& _trnset);

      virtual void
      train_epoch(size_t _epoch, const Dataset<InTyp,TgtTyp,ExemplarSeries>& _trnset);

      virtual void
      train_series(const ExemplarSeries<InTyp,TgtTyp>& _sample) = 0;

      virtual void
      train_exemplar(const Exemplar<InTyp,TgtTyp>& _exemplar);

      virtual void
      calc_weight_updates(const ValarrMap _egradient);

      double
      update_performance_traces(unsigned int _epoch, double _trnperf, TrainingRecord& _trec);

      void
      failback();

   private:
      void
      alloc();

   protected:
      NN<InTyp,TgtTyp>& nnet;

   private:
      FitFunc<TgtTyp> fitnessfunc;
      Eval<InTyp, TgtTyp, NN, Dataset, FitFunc> evaluator;

      DatasetTyp validation_dataset;
      DatasetTyp test_dataset;

      std::map<std::string, Array2D<double>> weight_updates;

      TrainingRecord training_record;
   };

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
   SupervisedTrainingAlgo<InTyp,
                          TgtTyp,
                          Sample,
                          NN,
                          Dataset,
                          Eval,
                          FitFunc,
                          LRPolicy>::SupervisedTrainingAlgo(NN<InTyp, TgtTyp>& _nnet) : LRPolicy(_nnet), nnet(_nnet)
   {
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & layers = nnet.get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;

         // Set to train layer biases by default.
         TrainerConfig::set_train_biases(id, true);
      }
   }

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
                          LRPolicy>::train(const DatasetTyp& _trnset)
   {
      //std::cout << "SupervisedTrainingAlgo.train()\n" << std::flush;

      double trn_perf, trn_stdev;
      double perf;

      alloc();

      size_t no_runs = TrainerConfig::training_runs();

      // If no randomization of training order is set then
      // make certain training set order is normalized now.
      if (!this->randomize_training_order())
         _trnset.normalize_order();

      for (size_t runndx = 0; runndx < no_runs; runndx++)
      {
         training_record.clear();

         if (runndx > 0)
            this->nnet.initialize_weights();

         save_network_weights(nnet, "initial_weights");

         /*
          * Evaluate and save the performance for the initial network
          */
         std::tie(trn_perf, trn_stdev) = evaluator.evaluate(this->nnet, _trnset);
         perf = update_performance_traces(0, trn_perf, training_record);

         std::cout << "SupervisedTrainingAlgo.train() initial perf : " << perf << "\n" << std::flush;

         training_record.best_epoch = 0;
         training_record.best_performance = perf;
         save_network_weights(nnet, "best_epoch");

         // *** train the network
         train_run(_trnset);

         const std::map<std::string, std::shared_ptr<NetworkLayer>>
            & layers = this->nnet.get_layers();
         for (auto& it : layers)
            training_record.network_weights[it.first] = it.second->weights();

         save_training_record(training_record);

         // TODO - update aggregate training statistics
      }
      //std::cout << "SupervisedTrainingAlgo.train() EXIT\n" << std::flush;
   }

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class SampleTyp,
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
   TrainingRecord
   SupervisedTrainingAlgo<InTyp,
                          TgtTyp,
                          SampleTyp,
                          NN,
                          Dataset,
                          Eval,
                          FitFunc,
                          LRPolicy>::train_run(const DatasetTyp& _trnset)
   {
      //std::cout << "SupervisedTrainingAlgo.train_one_run() ENTRY\n" << std::flush;

      double trn_perf, trn_stdev;
      double trn_perf_improvement;
      double perf;

      unsigned int failback_count = 0;

      LRPolicy::reset();

      // Previous performance vectorize - used for failback testing
      double prev_trn_perf = std::numeric_limits<double>::max();
      double failback_limit = TrainerConfig::error_increase_limit();

      // Init best performance assuming we are trying to minimize error
      training_record.stop_signal = TrainingStopSignal::UNKNOWN;

      // Iterate through training epochs
      size_t n_epochs = TrainerConfig::max_epochs();
      size_t epoch = 0;
      for (epoch = 0; epoch < n_epochs; epoch++)
      {
         std::cout << "SupervisedTrainingAlgo : epoch  " << epoch << "\n" << std::flush;

         // Save the network weights in case we need to fail back
         save_network_weights(nnet, "failback");

         // Call function to iterate over training samples and update
         // the network weights.
         train_epoch(epoch, _trnset);

         // Evaluate the performance of the updated network
         std::tie(trn_perf, trn_stdev) = evaluator.evaluate(this->nnet, _trnset);

         /*
          * If the performance on the training set worsens by an
          * amount greater than the fail-back limit then (1) restore
          * the previous weights, (2) lower the learning rates and
          * retry the epoch.
          */
         trn_perf_improvement =
            (prev_trn_perf > 0) ? (trn_perf - prev_trn_perf) / prev_trn_perf :
            (trn_perf - 1e-9) / 1e-9;
         if (trn_perf_improvement > failback_limit)
         {
            failback();
            epoch--;

            failback_count++;
            if (failback_count > this->max_failbacks())
            {
               training_record.stop_signal = TrainingStopSignal::MAX_FAILBACK_REACHED;
               return training_record;
            }
            continue;
         }
         else
         {
            failback_count = 0;
            LRPolicy::apply_learning_rate_adjustments();
         }

         // Update performance history in training record
         if (epoch < 10 || epoch % TrainerConfig::report_frequency() == 0
             || epoch == n_epochs - 1)
            perf = update_performance_traces(epoch + 1, trn_perf, training_record);

         // Call function to save network weights for the best epoch.
         if (perf < training_record.best_performance)
         {
            training_record.best_epoch = epoch + 1;
            training_record.best_performance = perf;
            if (perf < TrainerConfig::error_goal())
               training_record.stop_signal = TrainingStopSignal::CRITERIA_MET;

            save_network_weights(nnet, "best_epoch");
         }

         // If we've reached the target error goal then exit.
/*         if (perf < TrainerConfig::error_goal())
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

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class SampleTyp,
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
                          SampleTyp,
                          NN,
                          Dataset,
                          Eval,
                          FitFunc,
                          LRPolicy>::train_epoch(size_t _epoch, const Dataset<InTyp,TgtTyp,Exemplar>& _trnset)
   {
      //std::cout << "SupervisedTrainingAlgo.train_epoch(Exemplar)\n" << std::flush;

      bool pending_updates = false;

      if (this->randomize_training_order())
         _trnset.randomize_order();

      // Iterate through all samples in the training set_weights
      size_t sample_ndx = 0;
      for (auto& sample : _trnset)
      {
         train_exemplar(sample);
         pending_updates = true;

         // If training in online or mini-batch mode, update now.
         if (TrainerConfig::batch_mode() > 0
             && sample_ndx % TrainerConfig::batch_mode() == 0)
         {
            adjust_network_weights(nnet);
            pending_updates = false;
         }

         sample_ndx++;
      }

      // If training in batch mode or we there are updates pending from
      // an undersized mini-batch then update weights now
      if (TrainerConfig::batch_mode() == 0 || pending_updates)
         adjust_network_weights(nnet);
   }


   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class SampleTyp,
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
                          SampleTyp,
                          NN,
                          Dataset,
                          Eval,
                          FitFunc,
                          LRPolicy>::train_epoch(size_t _epoch, const Dataset<InTyp,TgtTyp,ExemplarSeries>& _trnset)
   {
      //std::cout << "SupervisedTrainingAlgo.train_epoch(ExemplarSeries)\n" << std::flush;

      bool pending_updates = false;

      if (this->randomize_training_order())
         _trnset.randomize_order();

      // Iterate through all samples in the training set_weights
      size_t sample_ndx = 0;
      for (auto& series : _trnset)
      {
         train_series(series);
         pending_updates = true;

         // If training in online or mini-batch mode, update now.
         if (TrainerConfig::batch_mode() > 0
             && sample_ndx % TrainerConfig::batch_mode() == 0)
         {
            adjust_network_weights(nnet);
            pending_updates = false;
         }

         sample_ndx++;
      }

      // If training in batch mode or there are updates pending from
      // an mini-batch then update weights now
      if (TrainerConfig::batch_mode() == 0 || pending_updates)
         adjust_network_weights(nnet);

      //std::cout << "SupervisedTrainingAlgo.train_epoch(ExemplarSeries) EXIT\n" << std::flush;
   }
   
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
      //std::cout << "SupervisedTrainerAlgo.train_exemplar()\n" << std::flush;

      const NNFeatureSet<TgtTyp>& nn_out = this->nnet.activate(_exemplar.first);

      //const std::map<std::string, std::valarray<double>>& nnoutv_map = nn_out.value_map();
      const std::map<std::string, std::valarray<double>>
         & targetv_map = _exemplar.second.value_map();

      ValarrMap gradient = evaluator.calc_error_gradient(_exemplar.second, nn_out);
      this->calc_weight_updates(gradient);
      LRPolicy::calc_learning_rate_adjustment(0);
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
                          LRPolicy>::train_series(const ExemplarSeries<InTyp,TgtTyp>& _series)
   {
      std::cout << "SupervisedTrainerAlgo.train_series()\n" << std::flush;

      // TODO - first pass at training series - fix this
      for (auto& exemplar : _series)
      {
         train_exemplar(exemplar);
      }
   }*/

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
   calc_weight_updates(const ValarrMap _egradient)
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
                          LRPolicy>::alloc()
   {
      weight_updates.clear();

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & layers = this->nnet.get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;
         const LayerWeights& w = it.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         weight_updates[id] = {};
         weight_updates[id].resize(dim.rows, dim.cols);
      }
   }

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
   double
   SupervisedTrainingAlgo<InTyp,
                          TgtTyp,
                          Sample,
                          NN,
                          Dataset,
                          Eval,
                          FitFunc,
                          LRPolicy>::update_performance_traces(unsigned int _epoch, double _trnperf, TrainingRecord& _trec)
   {
      double vld_perf, tst_perf;
      double vld_stdev, tst_stdev;

      double perf = _trnperf;

      // Save the training set performance for this epoch
      _trec.training_set_trace.push_back({.epoch=_epoch, .performance=_trnperf});

      // Record the performance on the validation set for this epoch
      if (validation_dataset.size() > 0)
      {
         std::tie(vld_perf, vld_stdev) =
            evaluator.evaluate(this->nnet, validation_dataset);

         // If validation set exist use it as the overall performance measure
         // in order to determine best weights using early stopping.
         perf = vld_perf;

         _trec.validation_set_trace.push_back({.epoch=_epoch, .performance=vld_perf});
      }

      // Record the performance on the test set for this epoch
      if (test_dataset.size() > 0)
      {
         std::tie(tst_perf, tst_stdev) = evaluator.evaluate(this->nnet, test_dataset);
         _trec.test_set_trace.push_back({.epoch=_epoch, .performance=tst_perf});
      }

      return perf;
   }

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
                          LRPolicy>::failback()
   {
      std::cout << "fail-back!!!!\n";
      restore_network_weights(nnet, "failback");
      LRPolicy::reduce_learning_rate();
   }
} // end namespace flexnnet


#endif //FLEX_NEURALNET_SUPERVISEDTRAININGALGO_H_
