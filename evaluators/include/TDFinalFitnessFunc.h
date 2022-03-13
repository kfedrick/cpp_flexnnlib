//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_TDFINALFITNESSFUNC_H_
#define FLEX_NEURALNET_TDFINALFITNESSFUNC_H_

#include "flexnnet.h"
#include "BaseFitnessFunc.h"

namespace flexnnet
{
   template<class OutTyp>
   class TDFinalFitnessFunc : public BaseFitnessFunc
   {
   public:

      void
      clear(void);

      void
      new_series(void);

      double
      calc_fitness(void);

      const ValarrMap&
      calc_error_gradient(const OutTyp& _target, const OutTyp& _nnout);

      const ValarrMap&
      calc_error_gradient(const OutTyp& _target, const OutTyp& _V_est, const OutTyp _prev_V_est);

      OutTyp
      calc_target(const OutTyp& _Vt, const OutTyp& _EVt) const;

   private:
      double sse, series_sse;
      size_t sample_count, series_len;
      ValarrMap td_gradient;
   };

   template<class OutTyp>
   inline
   void
   TDFinalFitnessFunc<OutTyp>::clear(void)
   {
      sse = 0;
      series_sse = 0;
      sample_count = 0;
      series_len = 0;
   }

   template<class OutTyp>
   inline
   void
   TDFinalFitnessFunc<OutTyp>::new_series(void)
   {
      sse += series_sse * series_sse / series_len;
      sample_count++;

      series_len = 0;
      series_sse = 0;
   }

   template<class OutTyp>
   const ValarrMap&
   TDFinalFitnessFunc<OutTyp>::calc_error_gradient(const OutTyp& _tgt, const OutTyp& _nnout)
   {
      //std::cout << "TDFinalFitnessFunc<OutTyp>::calc_error_gradient : ENTRY\n";

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
      TDFinalFitnessFunc::series_sse += sse;
      series_len++;

      //std::cout << "TDFinalFitnessFunc<OutTyp>::calc_error_gradient : RETURN\n";
      return td_gradient;
   }

   template<class OutTyp>
   inline
   const ValarrMap&
   TDFinalFitnessFunc<OutTyp>::calc_error_gradient(const OutTyp& _externR, const OutTyp& _V_est, const OutTyp _prev_V_est)
   {
      std::valarray<double> sqrdiff;

      const ValarrMap& externR_vamap = _externR.value_map();
      const ValarrMap& V_est_vamap = _V_est.value_map();
      const ValarrMap& prev_V_est_vamap = _prev_V_est.value_map();

      double sse = 0;
      for (const auto& externR : externR_vamap)
      {
         const std::string id = externR.first;

         if (externR.second.size() > 0)
            td_gradient[id] = externR.second - prev_V_est_vamap.at(id);
         else
            td_gradient[id] = V_est_vamap.at(id) - prev_V_est_vamap.at(id);

         sse += td_gradient[id].sum();
      }
      TDFinalFitnessFunc::series_sse += sse;
      series_len++;

      return td_gradient;
   }

   template<class OutTyp>
   inline
   double
   TDFinalFitnessFunc<OutTyp>::calc_fitness()
   {
      return (sample_count > 0) ? (sqrt(sse) / sample_count) : 0;
   }

   template<class OutTyp>
   inline
   OutTyp
   TDFinalFitnessFunc<OutTyp>::calc_target(const OutTyp& _Vt, const OutTyp& _EVt) const
   {
      //std::cout << "TDFinalFitnessFunc<OutTyp>::calc_target : ENTRY\n";

      ValarrMap tgt_vamap;

      const ValarrMap& Vt_vamap = _Vt.value_map();
      const ValarrMap& EVt_vamap = _EVt.value_map();

      int i = 0;
      for (const auto& EVt : EVt_vamap)
      {
         i++;

         const std::string id = EVt.first;
         if (Vt_vamap.at(id).size() > 0)
            tgt_vamap[id] = Vt_vamap.at(id);
         else
            tgt_vamap[id] = EVt.second;
      }

      OutTyp out = _Vt;
      out[0].decode(tgt_vamap.at("output"));

      //std::cout << "TDFinalFitnessFunc<OutTyp>::calc_target : about to EXIT\n";
      return out;
   }
}

#endif //FLEX_NEURALNET_TDFINALFITNESSFUNC_H_
