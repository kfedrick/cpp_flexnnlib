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
#include "BaseTrainingAlgo.h"
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
   template<class InTyp, class TgtTyp, template<class, class>
      class Sample,
      template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   class DeepRLAlgo : public BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>,
                      public TDTrainerConfig
   {
      using NNTyp = NeuralNet<InTyp, TgtTyp>;
      using DatasetTyp = DataSet<InTyp, TgtTyp, Sample>;
      using SampleTyp = Sample<InTyp, TgtTyp>;
      using ExemplarSeriesTyp = ExemplarSeries<InTyp, TgtTyp>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;

   public:
      DeepRLAlgo();

      void set_gamma(double _val);

   protected:
      virtual void train_sample(NNTyp& _nnet, const SampleTyp& _sample) = 0;

      /**
       * Calculate the neural network weight updates given the specified
       * output error gradient.
       *
       * Precondition:
       *    The network layers retain the state information
       *    for the input/target training exemplar used to generate
       *    the error gradient.
       *
       * @param _egradient
       */
      void calc_weight_updates(const NNTyp& _nnet, const ValarrMap& _tdgradient);

      void zero_eligibility_traces(const NNTyp& _nnet);

      void update_eligibility_traces(const NNTyp& _nnet);

      void alloc_working_memory(const NNTyp& _nnet) override;

   protected:
      // Cached layer states
      std::map<std::string, NetworkState> cached_layer_state;

      std::map<std::string, Array2D<double>> eligibility_trace;

      std::map<std::string, Array2D<double>> weight_updates;

   };

   template<class InTyp, class TgtTyp, template<class, class>
      class Sample, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   void DeepRLAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::set_gamma(double _val)
   {
      //evaluator.set_gamma(_val);
   }

   template<class InTyp, class TgtTyp, template<class, class>
      class Sample, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   DeepRLAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::DeepRLAlgo()
      : BaseTrainingAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>()
   {
      //alloc_working_memory(_nnet);
   }

   template<class InTyp, class TgtTyp, template<class, class>
      class Sample, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   void DeepRLAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::calc_weight_updates(
      const NNTyp& _nnet, const ValarrMap& _tdgradient)
   {
      //std::cout << "calc_weight_updates(" << _tderr << ")\n" << std::flush;
      double _tderr = _tdgradient.begin()->second[0];

      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = _nnet.get_layers();
      for (auto& it: layers)
      {
         std::string id = it.first;
         Array2D<double> lr = this->learning_rate_policy_obj.get_learning_rates(id);

         const Array2D<double> etrace_dEdw = this->eligibility_trace.at(id);

         const Array2D<double>::Dimensions dims = this->weight_updates[id].size();

         // If this layer doesn't train biases, stop before the last column
         unsigned int last_col = (TrainerConfig::train_biases(id)) ? dims.cols : dims.cols - 1;
         this->weight_updates[id] = 0;
         for (unsigned int row = 0; row < dims.rows; row++)
            for (unsigned int col = 0; col < last_col; col++)
               this->weight_updates[id].at(row, col) =
                  -lr.at(row, col) * _tderr * etrace_dEdw.at(row, col);

         this->accumulate_weight_updates(_nnet, id, this->weight_updates[id]);
      }
   }

   template<class InTyp, class TgtTyp, template<class, class>
      class Sample, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   void DeepRLAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::zero_eligibility_traces(const NNTyp& _nnet)
   {
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = _nnet.get_layers();
      for (auto& layer: network_layers)
         eligibility_trace[layer.first] = 0;
   }

   template<class InTyp, class TgtTyp, template<class, class>
      class Sample, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   void DeepRLAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::update_eligibility_traces(const NNTyp& _nnet)
   {
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = _nnet.get_layers();
      for (auto& layer: network_layers)
         eligibility_trace[layer.first] =
            layer.second->dEdw() + get_lambda() * eligibility_trace[layer.first];
   }

   template<class InTyp, class TgtTyp, template<class, class>
      class Sample, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   void DeepRLAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::alloc_working_memory(const NNTyp& _nnet)
   {
      std::cout << "BaseDeepRLAlgo::alloc_working_memory() ENTRY\n" << std::flush;

      eligibility_trace.clear();

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = _nnet.get_layers();
      for (auto& layer: network_layers)
         eligibility_trace[layer.first].set(layer.second->dEdw());

      weight_updates.clear();

      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = _nnet.get_layers();
      for (auto it: layers)
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
