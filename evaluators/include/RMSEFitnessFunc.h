//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_RMSEFITNESSFUNC_H_
#define FLEX_NEURALNET_RMSEFITNESSFUNC_H_

#include "flexnnet.h"
#include "BaseFitnessFunc.h"

namespace flexnnet
{
   template<class _OutTyp>
   class RMSEFitnessFunc : public BaseFitnessFunc
   {
   public:
      void clear(void);
      double calc_fitness(void);
      const ValarrMap& calc_error_gradient(const _OutTyp& _target, const _OutTyp& _nnout);

   public:
      double sse;
      size_t sample_count;
      ValarrMap egradient;
   };

   template<class _OutTyp>
   void RMSEFitnessFunc<_OutTyp>::clear(void)
   {
      sse = 0;
      sample_count = 0;
   }

   template<class _OutTyp>
   const ValarrMap&  RMSEFitnessFunc<_OutTyp>::calc_error_gradient(const _OutTyp& _target, const _OutTyp& _actual)
   {
      static std::valarray<double> sqrdiff;

      const ValarrMap& tgt_va = _target.value_map();
      const ValarrMap& act_va = _actual.value_map();

      double sse = 0;
      for (const auto& it : tgt_va)
      {
         egradient[it.first] = -(it.second - act_va.at(it.first));
         sqrdiff = egradient[it.first] * egradient[it.first];

         sse += sqrdiff.sum();
      }

      RMSEFitnessFunc::sse += sse;
      sample_count++;

      return egradient;
   }

   template<class _OutTyp>
   double RMSEFitnessFunc<_OutTyp>::calc_fitness()
   {
      return (sample_count>0) ? (sqrt(0.5*sse/sample_count)) : 0;
   }
}

#endif //FLEX_NEURALNET_RMSEFITNESSFUNC_H_
