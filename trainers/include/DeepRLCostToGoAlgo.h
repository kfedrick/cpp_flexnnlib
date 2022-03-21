//
// Created by kfedrick on 3/15/22.
//

#ifndef FLEX_NEURALNET_DEEPRLCOSTTOGOALGO_H_
#define FLEX_NEURALNET_DEEPRLCOSTTOGOALGO_H_

#include <DeepRLAlgo.h>

namespace flexnnet
{
   template<class InTyp, class TgtTyp, template<class, class>
      class Sample, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   class DeepRLCostToGoAlgo : public DeepRLAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>
   {
      using NNTyp = NeuralNet <InTyp, TgtTyp>;
      using SampleTyp = Sample<InTyp, TgtTyp>;
      using ExemplarSeriesTyp = ExemplarSeries<InTyp, TgtTyp>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;

   public:
      DeepRLCostToGoAlgo();

      void train_sample(NNTyp& _nnet, const ExemplarSeriesTyp& _series);
   };

   template<class InTyp, class TgtTyp, template<class, class>
      class Sample, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   DeepRLCostToGoAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>::DeepRLCostToGoAlgo() : DeepRLAlgo<InTyp, TgtTyp, Sample, FitFunc, LRPolicy>()
   {

   }

   template<class InTyp, class TgtTyp, template<class, class>
      class ExemplarSeriesTyp, template<class, class, template<class, class> class>
      class FitFunc, class LRPolicy>
   void DeepRLCostToGoAlgo<InTyp, TgtTyp, ExemplarSeriesTyp, FitFunc, LRPolicy>::train_sample(NNTyp& _nnet,
                                                                                              const ExemplarSeriesTyp& _series)
   {
      //std::cout << "DeepRLCostToGoAlgo.train_sample()\n" << std::flush;

      // If series is less than 2 steps long then there's nothing to learn
      if (_series.size() < 2)
         return;

      ValarrMap td_gradient = _series[0].second.value_map();

      // Create "error" of -1 to allow calculating gradients wrt Y
      ValarrMap ones = _nnet.value_map();
      for (auto& it: ones)
         it.second = -1.0;

      /*
       * Present the first item in the series and calculate the
       * initial eligibility trace info
       */
      _nnet.activate(_series[0].first);

      this->zero_eligibility_traces(_nnet);
      _nnet.backprop(ones);
      this->update_eligibility_traces(_nnet);

      /*
       * Present the exemplars for the remaining exemplars prior
       * to the terminal state.
       */
      int series_len = _series.size();

      // try training terminal state
      for (int ndx = 1; ndx < series_len - 1; ndx++)
      {
         // Save vectorize estimate from step t-1
         TgtTyp nnout0 = _nnet.value();

         // Activate the network for exemplar at time t
         const ExemplarTyp& exemplar = _series[ndx];
         const TgtTyp& nnout1 = _nnet.activate(exemplar.first);

         // Train the network using the target and vectorize estimates
         //const std::valarray<double>& targetv = exemplar.second.vectorizeee();

         //TgtTyp tgt = evaluator.calc_target(exemplar.second, nnout1);
         //ValarrMap td_gradientset = evaluator.calc_error_gradient(tgt, nnout0);
         //---evaluator.calc_error_gradient(exemplar.second, nnout1, nnout0, td_gradientset);
         this->fitnessfunc.calc_tde_gradient(exemplar.second, nnout1, nnout0, td_gradient);

         this->calc_weight_updates(_nnet, td_gradient);

         // Update the eligibility traces
         _nnet.backprop(ones);
         this->update_eligibility_traces(_nnet);
      }
      const ExemplarTyp& last_exemplar = _series[series_len - 1];

      // Assign last examplar to zeronnout just to get the feature names
      TgtTyp zeronnout = last_exemplar.second;
      ValarrMap vm({{"output", {0}}});
      std::valarray<double> va = {0};
      zeronnout[0].decode(va);

      // Train the network on the terminal state.
      //const std::valarray<double>& targetv = _series[series_len - 1].second.vectorizeee();
      TgtTyp nnout0 = _nnet.value();

      // second training on terminal state
      //TgtTyp tgt = evaluator.calc_target(last_exemplar.second, zeronnout);
      //TgtTyp tgt = evaluator.calc_target(last_exemplar.second, nnout0);
      //ValarrMap td_gradientset = evaluator.calc_error_gradient(tgt, nnout0);
      //--evaluator.calc_error_gradient(last_exemplar.second, zeronnout, nnout0, td_gradientset);
      this->fitnessfunc.calc_tde_gradient(last_exemplar.second, zeronnout, nnout0, td_gradient);

      this->calc_weight_updates(_nnet, td_gradient);

      //std::cout << "DeepRLAlgo.train_sample() EXIT\n" << std::flush;
   }

}

#endif // FLEX_NEURALNET_DEEPRLCOSTTOGOALGO_H_
