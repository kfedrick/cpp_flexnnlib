//
// Created by kfedrick on 3/4/21.
//

#ifndef FLEX_NEURALNET_DERIVEDEVALUATOR_H_
#define FLEX_NEURALNET_DERIVEDEVALUATOR_H_

#include "Evaluator.h"
#include "FitnessFunction.h"

namespace flexnnet
{
   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet>
   class DerivedEvaluator : public Evaluator<_InTyp, _OutTyp, _NN, _DataSet, FitnessFunction>
   {
      using _BaseEval = Evaluator<_InTyp, _OutTyp, _NN, _DataSet, FitnessFunction>;
      using _NNTyp = _NN<_InTyp, _OutTyp>;
      using _DatasetTyp = _DataSet<_InTyp, _OutTyp>;

   public:
      DerivedEvaluator(const _DatasetTyp& _tstset);
      double evaluate(_NNTyp& _nnet, const _DatasetTyp& _tstset);
      void doit(void);
   };

   template<class _In, class _Out, template<class, class> class _NN, template<class, class> class _Dataset>
   DerivedEvaluator<_In, _Out, _NN, _Dataset>::DerivedEvaluator(const _DatasetTyp& _tstset) : Evaluator<_In, _Out, _NN, _Dataset, FitnessFunction>::Evaluator(_tstset)
   {
      std::cout << "DerivedEvaluator::DerivedEvaluator()\n";
   }

   template<class _In, class _Out, template<class, class> class _NN, template<class, class> class _Dataset>
   double DerivedEvaluator<_In, _Out, _NN, _Dataset>::evaluate(_NNTyp& _nnet, const _DatasetTyp& _tstset)
   {
      std::cout << "DerivedEvaluator::evaluate()\n";

      // Vector to hold performance results for each sampling
      size_t scount = _BaseEval::sampling_count();
      std::valarray<double> perf(scount);

      // Iterate through all exemplars in the training set_weights
      for (size_t sample_ndx = 0; sample_ndx < scount; sample_ndx++)
      {
         // TODO - generate sampling mask and pass to evaluate_subsampling
         perf[sample_ndx] = _BaseEval::evaluate_subsampling(sample_ndx, _nnet, _tstset);
      }

      // TODO - calculate statistics across samplings

      return perf.sum() / scount;
   }

   template<class _In, class _Out, template<class, class> class _NN, template<class, class> class _Dataset>
   void DerivedEvaluator<_In, _Out, _NN, _Dataset>::doit(void)
   {
   }

}

#endif //FLEX_NEURALNET_DERIVEDEVALUATOR_H_
