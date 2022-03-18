//
// Created by kfedrick on 3/15/22.
//

#ifndef FLEX_NEURALNET_DEEPRLFINALCOSTALGO_H_
#define FLEX_NEURALNET_DEEPRLFINALCOSTALGO_H_

#include <DeepRLAlgo.h>

namespace flexnnet
{
   template<class InTyp, class TgtTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class> class FitFunc,

      class LRPolicy

   >

   class DeepRLFinalCostAlgo : public DeepRLAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>
   {
      using ExemplarSeriesTyp = ExemplarSeries<InTyp, TgtTyp>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;

   public:
      DeepRLFinalCostAlgo(NN<InTyp, TgtTyp>& _nnet);

      void train_sample(const ExemplarSeriesTyp& _series);
   };

   template<class InTyp, class TgtTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class>
      class FitFunc,

      class LRPolicy

   >

   DeepRLFinalCostAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>::DeepRLFinalCostAlgo(
      NN<InTyp, TgtTyp>& _nnet) : DeepRLAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>(_nnet)
   {

   }

   template<class InTyp, class TgtTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class>
      class FitFunc,

      class LRPolicy

   >

   void DeepRLFinalCostAlgo<InTyp, TgtTyp, NN, Dataset, FitFunc, LRPolicy>::train_sample(
      const ExemplarSeriesTyp& _series)
   {
      //std::cout << "DeepRLFinalCostAlgo.train_sample()\n" << std::flush;

      ValarrMap td_gradient = _series[0].second.value_map();

      // Create "error" of 1 to allow calculating gradients wrt Y
      ValarrMap ones = this->nnet.value_map();
      for (auto& it: ones)
         it.second = -1.0;

      /*
       * Present the first item in the series and calculate the
       * initial eligibility trace info
       */
      this->nnet.activate(_series[0].first);

      this->nnet.backprop(ones);

      this->zero_eligibility_traces(this->nnet);
      this->update_eligibility_traces(this->nnet);

      /*
       * Present the exemplars for the remaining exemplars prior
       * to the terminal state.
       */
      int series_len = _series.size();
      for (int ndx = 1; ndx < series_len; ndx++)
      {
         // Save vectorize estimate
         TgtTyp nnout0 = this->nnet.value();

         // Activate the network for the next exemplar
         const ExemplarTyp& exemplar = _series[ndx];
         const TgtTyp& nnout1 = this->nnet.activate(exemplar.first);

         if (exemplar.valid_target())
         {
            //std::cout << "DeepRLFinalCostAlgo.train_sample() valid target\n" << std::flush;
            //--this->evaluator.calc_error_gradient(exemplar.second, nnout0, td_gradient);
            this->fitnessfunc.calc_tde_gradient(exemplar.second, nnout0, td_gradient);

         }
         else
            //--this->evaluator.calc_error_gradient(nnout1, nnout0, td_gradient);
            this->fitnessfunc.calc_tde_gradient(nnout1, nnout0, td_gradient);

         this->calc_weight_updates(this->nnet, td_gradient);
         //std::cout << "DeepRLFinalCostAlgo.train_sample() egrad = " << td_gradient["output"][0] << "\n" << std::flush;

         // Update the eligibility traces
         this->nnet.backprop(ones);
         this->update_eligibility_traces(this->nnet);

         //this->evaluator.new_series();
      }
      //std::cout << "DeepRLFinalCostAlgo.train_sample() EXIT\n" << std::flush;

   }
}

#endif // FLEX_NEURALNET_DEEPRLFINALCOSTALGO_H_
