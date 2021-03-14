//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_FITNESSFUNCTION_H_
#define FLEX_NEURALNET_FITNESSFUNCTION_H_

#include "flexnnet.h"

namespace flexnnet
{
   template<class _OutTyp>
   class FitnessFunction
   {
   public:
      void reset(void);
      void analyze_sample(const _OutTyp& _target, const _OutTyp& _actual);
      double calc_fitness(void);
      const ValarrMap& calc_error_gradient(const _OutTyp& _target, const _OutTyp& _nnout) {};

   public:
      double sse;
   };

   template<class _OutTyp>
   void FitnessFunction<_OutTyp>::reset(void)
   {
      sse = 0;
   }

   template<class _OutTyp>
   void FitnessFunction<_OutTyp>::analyze_sample(const _OutTyp& _target, const _OutTyp& _actual)
   {
      std::cout << "FitnessFunction::analyze_sample()\n" << std::flush;
      sse += 1;
   }

   template<class _OutTyp>
   double FitnessFunction<_OutTyp>::calc_fitness()
   {
      return sse;
   }
}

#endif //FLEX_NEURALNET_FITNESSFUNCTION_H_
