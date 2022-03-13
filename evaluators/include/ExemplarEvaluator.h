//
// Created by kfedrick on 3/12/22.
//

#ifndef FLEX_NEURALNET_EXEMPLAREVALUATOR_H_
#define FLEX_NEURALNET_EXEMPLAREVALUATOR_H_

#include <iostream>
#include <flexnnet.h>
#include <Exemplar.h>
#include <ExemplarSeries.h>
#include <BasicEvalConfig.h>
#include "Reinforcement.h"
#include <BaseEvaluator.h>

namespace flexnnet
{
   template<class InTyp, class OutTyp,
      template<class, class> class NN,
      template<class, class, template<class, class> class> class _DataSet,
      template<class> class FitnessFunc>
   //class ExemplarEvaluator : public BasicEvalConfig, public FitnessFunc<OutTyp>
   class ExemplarEvaluator : public BaseEvaluator<InTyp, OutTyp, Exemplar, NN, _DataSet, FitnessFunc>
   {
      using NNTyp = NN<InTyp, OutTyp>;
      using ExemplarTyp = Exemplar<InTyp, OutTyp>;

   public:
      ExemplarEvaluator();

      void
      evaluate_sample(size_t _s_index, NNTyp& _nnet, const ExemplarTyp& _series);
   };

   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   ExemplarEvaluator<InTyp,
               OutTyp,
               NN,
               Dataset,
               FitnessFunc>::ExemplarEvaluator()
   {
   }

   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   void
   ExemplarEvaluator<InTyp,
               OutTyp,
               NN,
               Dataset,
               FitnessFunc>::evaluate_sample(size_t _s_index, NNTyp& _nnet, const ExemplarTyp& _exemplar)
   {
      std::cout << "ExemplarEvaluator::evaluate_exemplar()\n";
      const OutTyp& nnout = _nnet.activate(_exemplar.first);
      FitnessFunc<OutTyp>::calc_error_gradient(_exemplar.second, nnout);
   }


}


#endif // FLEX_NEURALNET_EXEMPLAREVALUATOR_H_
