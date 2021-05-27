//
// Created by kfedrick on 4/19/21.
//

#ifndef FLEX_NEURALNET_DEEPRLALGO_H_
#define FLEX_NEURALNET_DEEPRLALGO_H_

#include <flexnnet.h>
#include "DataSet.h"
#include "TDTrainerConfig.h"
#include "TrainingRecord.h"
#include "TrainingReport.h"
#include "BaseTrainer.h"
#include "TDEvaluator.h"
#include "SupervisedTrainingAlgo.h"
#include "TDFinalFitnessFunc.h"
#include "Reinforcement.h"

namespace flexnnet
{
   /**
    * Deep NetworkReinforcement Learning Algorithm
    *
    * @tparam InTyp - Neural network input data typename
    * @tparam TgtTyp - Neural net training set target data typename
    * @tparam NN - Neural network class name
    * @tparam Dataset - Dataset container class name
    * @tparam FitFunc - Fitness function class name
    * @tparam LRPolicy - Learning rate policy class name
    */
   template<class InTyp, class TgtTyp,
      template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitFunc,
      class LRPolicy>
   class DeepRLAlgo : public SupervisedTrainingAlgo<InTyp,
                                                    TgtTyp,
                                                    ExemplarSeries,
                                                    NN,
                                                    Dataset,
                                                    TDEvaluator,
                                                    FitFunc,
                                                    LRPolicy>,
                      public TDTrainerConfig
   {
      using DatasetTyp = Dataset<InTyp, TgtTyp, ExemplarSeries>;
      using ExemplarSeriesTyp = ExemplarSeries<InTyp, TgtTyp>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;

   public:
      DeepRLAlgo(NN<InTyp, TgtTyp>& _nnet);
      void set_gamma(double _val);

   protected:
      void
      train_series(const ExemplarSeries<InTyp, TgtTyp>& _sample);

      void
      train_series_final_cost(const ExemplarSeriesTyp& _series);


      void
      train_series_cost_to_go(const ExemplarSeriesTyp& _series);

      void
      train_cost_to_go(const std::valarray<double>& _V, const std::valarray<double>& _Re1, const std::valarray<double>& _Re0);

      void
      train_final_cost(const std::valarray<double>& _V, const std::valarray<double>& _Re1, const std::valarray<double>& _Re0);

      void
      calc_weight_updates(const FeatureVector& _tdgradient);

      void
      zero_eligibility_traces();

      void
      update_eligibility_traces();

      void alloc();

   private:
      TDEvaluator<InTyp, TgtTyp, NN, Dataset, FitFunc> evaluator;

      // Cached layer states
      std::map<std::string, NetworkState> cached_layer_state;

      std::map<std::string, Array2D<double>> eligibility_trace;

      std::map<std::string, Array2D<double>> weight_updates;

   };

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void DeepRLAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>::set_gamma(double _val)
   {
      evaluator.set_gamma(_val);
   }

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   DeepRLAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>::DeepRLAlgo(NN<InTyp,
                                                                            TgtTyp>& _nnet)
      : SupervisedTrainingAlgo<InTyp,
                               TgtTyp,
                               ExemplarSeries,
                               NN,
                               Dataset,
                               TDEvaluator,
                               FitFunc,
                               LRPolicy>(_nnet)
   {
      alloc();
   }

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp,
              TgtTyp,
              NN,
              Dataset,
              FitFunc,
              LRPolicy>::train_series(const ExemplarSeriesTyp& _series)
   {
      //std::cout << "DeepRLAlgo.train_series()\n" << std::flush;
      if (get_td_mode() == FINAL_COST)
         train_series_final_cost(_series);
      else
         train_series_cost_to_go(_series);

      //std::cout << "DeepRLAlgo.train_series() EXIT\n" << std::flush;

   }

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp,
              TgtTyp,
              NN,
              Dataset,
              FitFunc,
              LRPolicy>::train_series_final_cost(const ExemplarSeriesTyp& _series)
   {
      //std::cout << "DeepRLAlgo.train_series_final_cost()\n" << std::flush;
      //std::valarray<double> Re0(this->nnet.size());
      //const std::valarray<double> zero_Re(0.0, this->nnet.size());

      // Create "error" of 1 to allow calculating gradients wrt Y
      ValarrMap ones = this->nnet.value_map();
      for (auto& it : ones)
         it.second = -1.0;

      /*
       * Present the first item in the series and calculate the
       * initial eligibility trace info
       */
      this->nnet.activate(_series[0].first);

      this->nnet.backprop(ones);

      zero_eligibility_traces();
      update_eligibility_traces();

      /*
       * Present the exemplars for the remaining exemplars prior
       * to the terminal state.
       */
      int series_len = _series.size();
      for (int ndx=1; ndx < series_len; ndx++)
      {
         // Save vectorize estimate
         //Re0 = this->nnet.vectorize().vectorize();
         TgtTyp nnout0 = this->nnet.vectorize_features();

         // Activate the network for the next exemplar
         const ExemplarTyp& exemplar = _series[ndx];
         const TgtTyp& nnout1 = this->nnet.activate(exemplar.first);

         // Train the network using the target and vectorize estimates
         const std::valarray<double>& targetv = exemplar.second.vectorize_features();
         //train_final_cost(targetv, nnout1.vectorize(), Re0);

         TgtTyp tgt = evaluator.calc_target(exemplar.second, nnout1);
         FeatureVector td_gradient = evaluator.calc_error_gradient(tgt, nnout0);
         calc_weight_updates(td_gradient);

         // Update the eligibility traces
         this->nnet.backprop(ones);
         update_eligibility_traces();
      }
      //std::cout << "DeepRLAlgo.train_series_final_cost() EXIT\n" << std::flush;

   }

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp,
              TgtTyp,
              NN,
              Dataset,
              FitFunc,
              LRPolicy>::train_series_cost_to_go(const ExemplarSeriesTyp& _series)
   {
      //std::cout << "DeepRLAlgo.train_series_cost_to_go()\n" << std::flush;

      // If series is less than 2 steps long then there's nothing to learn
      if (_series.size() < 2)
         return;

      std::valarray<double> Re0;
      const std::valarray<double> zero_Re(0.0, this->nnet.size());

      // Create "error" of 1 to allow calculating gradients wrt Y
      ValarrMap ones = this->nnet.value_map();
      for (auto& it : ones)
         it.second = -1.0;

      /*
       * Present the first item in the series and calculate the
       * initial eligibility trace info
       */
      this->nnet.activate(_series[0].first);

      zero_eligibility_traces();
      this->nnet.backprop(ones);
      update_eligibility_traces();

      /*
       * Present the exemplars for the remaining exemplars prior
       * to the terminal state.
       */
      int series_len = _series.size();
      for (int ndx=1; ndx < series_len-1; ndx++)
      {
         // Save vectorize estimate from step t-1
         Re0 = this->nnet.vectorize_features().vectorize_features();
         TgtTyp nnout0 = this->nnet.vectorize_features();

         // Activate the network for exemplar at time t
         const ExemplarTyp& exemplar = _series[ndx];
         const TgtTyp& nnout1 = this->nnet.activate(exemplar.first);

         // Train the network using the target and vectorize estimates
         const std::valarray<double>& targetv = exemplar.second.vectorize_features();

         TgtTyp tgt = evaluator.calc_target(exemplar.second, nnout1);
         FeatureVector td_gradient = evaluator.calc_error_gradient(tgt, nnout0);
         calc_weight_updates(td_gradient);

         // Update the eligibility traces
         this->nnet.backprop(ones);
         update_eligibility_traces();
      }

      // Train the network on the terminal state.
      const std::valarray<double>& targetv = _series[series_len - 1].second.vectorize_features();
      TgtTyp nnout0 = this->nnet.vectorize_features();
      FeatureVector td_gradient = evaluator.calc_error_gradient(_series[series_len - 1].second, nnout0);
      calc_weight_updates(td_gradient);

      //std::cout << "DeepRLAlgo.train_series_cost_to_go() EXIT\n" << std::flush;

   }

/*   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp,
              TgtTyp,
              NN,
              Dataset,
              FitFunc,
              LRPolicy>::train_cost_to_go(const std::valarray<double>& _V, const std::valarray<double>& _Re1, const std::valarray<double>& _Re0)
   {
      //std::cout << "calc td error " << _V[0] << " " << _Re0[0] << " " << _Re1[0] << "\n" << std::flush;
      double td_error = _V[0] + get_gamma() * _Re1[0] - _Re0[0];
      FeatureVector tdgradient;
      tdgradient["output"] = {td_error};
      calc_weight_updates(tdgradient);
   }

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp,
              TgtTyp,
              NN,
              Dataset,
              FitFunc,
              LRPolicy>::train_final_cost(const std::valarray<double>& _V, const std::valarray<double>& _Re1, const std::valarray<double>& _Re0)
   {
      //std::cout << "DeepRLAlgo.train_final_cost() " << _Re0[0] << " " << _Re1[0] << "\n" << std::flush;
      //if (_V.size() > 0)
      //   std::cout << "DeepRLAlgo.train_final_cost() V=" << _V[0] << "\n" << std::flush;

      double td_error;

      // If there is a external reinforcement specified vectorize, use it. Otherwise
      // use the current network estimate as the target vectorize function.
      if (_V.size() > 0)
         td_error = _V[0] - _Re0[0];
      else
         td_error = _Re1[0] - _Re0[0];


      //calc_weight_updates(td_error);
   }*/

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>::calc_weight_updates(const FeatureVector& _tdgradient)
   {
      //std::cout << "calc_weight_updates(" << _tderr << ")\n" << std::flush;
      double _tderr = _tdgradient.value()[0];

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & layers = this->nnet.get_layers();
      for (auto& it : layers)
      {
         std::string id = it.first;
         Array2D<double> lr = LRPolicy::get_learning_rates(id);

         const Array2D<double> etrace_dEdw = eligibility_trace.at(id);

         const Array2D<double>::Dimensions dims = weight_updates[id].size();

         // If this layer doesn't train biases, stop before the last column
         unsigned int
            last_col = (TrainerConfig::train_biases(id)) ? dims.cols : dims.cols - 1;
         weight_updates[id] = 0;
         for (unsigned int row = 0; row < dims.rows; row++)
            for (unsigned int col = 0; col < last_col; col++)
               weight_updates[id].at(row, col) = -lr.at(row, col) * _tderr * etrace_dEdw.at(row, col);

         this->accumulate_weight_updates(id, weight_updates[id]);
      }
   }

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>::zero_eligibility_traces()
   {
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = this->nnet.get_layers();
      for (auto& layer : network_layers)
         eligibility_trace[layer.first] = 0;
   }

   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>::update_eligibility_traces()
   {
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = this->nnet.get_layers();
      for (auto& layer : network_layers)
         eligibility_trace[layer.first] =
            layer.second->dEdw() + get_lambda() * eligibility_trace[layer.first];
   }


   template<class InTyp,
      class TgtTyp,
      template<class, class>
      class NN,
      template<class, class, template<class, class> class>
      class Dataset,
      template<class>
      class FitFunc,
      class LRPolicy>
   void
   DeepRLAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>::alloc()
   {
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = this->nnet.get_layers();
      for (auto& layer : network_layers)
         eligibility_trace[layer.first].set(layer.second->dEdw());

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
} // end namespace flexnnet

#endif //FLEX_NEURALNET_DEEPRL_H_
