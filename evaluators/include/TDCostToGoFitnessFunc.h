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
      static constexpr double DEFAULT_LAMBDA = 0.3;

   public:
      void set_gamma(double _val);
      void set_lambda(double _val);

      void
      clear(void);

      void
      new_series(void);

      double
      calc_fitness(void);

      const ValarrMap&
      calc_error_gradient(const _OutTyp& _externR, const _OutTyp& _V_est, const _OutTyp _prev_V_est);

      const ValarrMap&
      calc_error_gradient(const _OutTyp& _target, const _OutTyp& _nnout);

      _OutTyp
      calc_target(const _OutTyp& _Vt, const _OutTyp& _EVt) const;

   private:
      double gamma{DEFAULT_GAMMA};
      double lambda{DEFAULT_LAMBDA};

      std::vector<double> tde_vec;

      double sse, series_sse, max_EV;
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
            << "Error : TDCostToGoFitnessFunc.set_gamma() - illegal vectorize" << _gamma << "\n";
         throw std::invalid_argument(err_str.str());
      }

      gamma = _gamma;
   }

   template<class OutTyp>
   void
   TDCostToGoFitnessFunc<OutTyp>::set_lambda(double _lambda)
   {
      if (_lambda < 0 || _lambda > 1.0)
      {
         std::ostringstream err_str;
         err_str
            << "Error : TDCostToGoFitnessFunc.set_lambda() - illegal vectorize" << _lambda << "\n";
         throw std::invalid_argument(err_str.str());
      }

      lambda = _lambda;
   }

   template<class OutTyp>
   void
   TDCostToGoFitnessFunc<OutTyp>::clear(void)
   {
      sse = 0;
      sample_count = 0;
      series_sse = 0;
      series_len = 0;
      max_EV = 1.0;
      tde_vec.clear();
   }

   template<class OutTyp>
   void
   TDCostToGoFitnessFunc<OutTyp>::new_series(void)
   {
      //sse += series_sse * series_sse / (series_len+1);
      //sse += series_sse;
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

      double tde = 0;
      for (const auto& tgt : tgt_vamap)
      {
         const std::string id = tgt.first;
         td_gradient[id] = tgt.second - nnout_vamap.at(id);
         tde += td_gradient[id].sum();

         if (nnout_vamap.at(id).max() > max_EV)
            max_EV = nnout_vamap.at(id).max();
      }
      tde_vec.push_back(tde);
      TDCostToGoFitnessFunc::series_sse = (tde * tde) + TDCostToGoFitnessFunc::series_sse;
      series_len++;

      return td_gradient;
   }

   template<class OutTyp>
   inline
   const ValarrMap&
   TDCostToGoFitnessFunc<OutTyp>::calc_error_gradient(const OutTyp& _externR, const OutTyp& _V_est, const OutTyp _prev_V_est)
   {
      const ValarrMap& externR_vamap = _externR.value_map();
      const ValarrMap& V_est_vamap = _V_est.value_map();
      const ValarrMap& prev_V_est_vamap = _prev_V_est.value_map();

      double tde = 0;
      for (const auto& externR : externR_vamap)
      {
         const std::string id = externR.first;
         td_gradient[id] = externR.second + gamma * _V_est.at(id) - _prev_V_est.at(id);
         tde += td_gradient[id].sum();
      }
      TDCostToGoFitnessFunc::series_sse = tde + TDCostToGoFitnessFunc::series_sse;
      series_len++;

      return td_gradient;
   }

   template<class _OutTyp>
   double
   TDCostToGoFitnessFunc<_OutTyp>::calc_fitness()
   {
      //return (sample_count > 0) ? (sse / sample_count) : 0;

      // Using series_len as sum of lengths of all series
      //return (series_len > 0) ? (sse / series_len) : 0;

      sse = 0;
      for (double tde : tde_vec)
      {
         sse += (tde/max_EV * tde/max_EV);
      }

      return (tde_vec.size() > 0) ? (sse / tde_vec.size()) : 0;
   }

   template<class OutTyp>
   OutTyp
   TDCostToGoFitnessFunc<OutTyp>::calc_target(const OutTyp& _Vt, const OutTyp& _EVt) const
   {
      ValarrMap tgt_vamap;

      const ValarrMap& Vt_vamap = _Vt.value_map();
      const ValarrMap& EVt_vamap = _EVt.value_map();

      for (const auto& Vt : Vt_vamap)
      {
         const std::string id = Vt.first;
         tgt_vamap[id] = Vt.second + gamma * EVt_vamap.at(id);
      }

      OutTyp out = _Vt;
      out[0].decode(tgt_vamap.at("output"));
      return out;

      //return OutTyp(tgt_vamap);
   }
}

#endif //FLEX_NEURALNET_TDCOSTTOGOFITNESSFUNC_H_
