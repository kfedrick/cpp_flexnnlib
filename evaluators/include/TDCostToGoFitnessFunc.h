//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_TDCOSTTOGOFITNESSFUNC_H_
#define FLEX_NEURALNET_TDCOSTTOGOFITNESSFUNC_H_

#include "flexnnet.h"
#include "BaseFitnessFunc.h"

namespace flexnnet
{
   template<class _OutTyp>
   class TDCostToGoFitnessFunc : public BaseFitnessFunc
   {
   public:
      static constexpr double DEFAULT_GAMMA = 0.9;

   public:
      void set_gamma(double _val);

      void
      clear(void);

      void
      new_series(void);

      double
      calc_fitness(void);

      const ValarrMap&
      calc_error_gradient(const _OutTyp& _V, const _OutTyp& _Re1, const _OutTyp& _Re0);

      const ValarrMap&
      calc_error_gradient(const _OutTyp& _target, const _OutTyp& _nnout);

      _OutTyp
      calc_target(const _OutTyp _Vt, const _OutTyp _EVt) const;

   private:
      double gamma{DEFAULT_GAMMA};
      double sse, series_sse;
      size_t sample_count, series_len;
      ValarrMap td_gradient;
   };

   template<class OutTyp>
   void
   TDCostToGoFitnessFunc<OutTyp>::set_gamma(double _gamma)
   {
      if (_gamma < 0 || _gamma > 1.0)
      {
         std::ostringstream err_str;
         err_str
            << "Error : TDCostToGoFitnessFunc.set_gamma() - illegal value" << _gamma << "\n";
         throw std::invalid_argument(err_str.str());
      }

      gamma = _gamma;
   }

   template<class OutTyp>
   void
   TDCostToGoFitnessFunc<OutTyp>::clear(void)
   {
      sse = 0;
      sample_count = 0;
      series_sse = 0;
      series_len = 0;
   }

   template<class OutTyp>
   void
   TDCostToGoFitnessFunc<OutTyp>::new_series(void)
   {
      sse += series_sse * series_sse / (series_len+1);
      sample_count++;

      series_len = 0;
      series_sse = 0;
   }

   template<class OutTyp>
   const ValarrMap&
   TDCostToGoFitnessFunc<OutTyp>::calc_error_gradient(const OutTyp& _tgt, const OutTyp& _nnout)
   {
      std::valarray<double> sqrdiff;

      const ValarrMap& tgt_vamap = _tgt.value_map();
      const ValarrMap& nnout_vamap = _nnout.value_map();

      double sse = 0;
      for (const auto& tgt : tgt_vamap)
      {
         const std::string id = tgt.first;
         td_gradient[id] = tgt.second - nnout_vamap.at(id);
         sse += td_gradient[id].sum();
      }
      TDCostToGoFitnessFunc::series_sse = sse + TDCostToGoFitnessFunc::series_sse;
      series_len++;

      return td_gradient;
   }

   template<class _OutTyp>
   double
   TDCostToGoFitnessFunc<_OutTyp>::calc_fitness()
   {
      return (sample_count > 0) ? (sse / sample_count) : 0;
   }

   template<class OutTyp>
   OutTyp
   TDCostToGoFitnessFunc<OutTyp>::calc_target(const OutTyp _Vt, const OutTyp _EVt) const
   {
      ValarrMap tgt_vamap;

      const ValarrMap& Vt_vamap = _Vt.value_map();
      const ValarrMap& EVt_vamap = _EVt.value_map();

      for (const auto& Vt : Vt_vamap)
      {
         const std::string id = Vt.first;
         tgt_vamap[id] = Vt.second + gamma * EVt_vamap.at(id);
      }

      return OutTyp(tgt_vamap);
   }
}

#endif //FLEX_NEURALNET_TDCOSTTOGOFITNESSFUNC_H_
