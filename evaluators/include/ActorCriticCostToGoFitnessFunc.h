//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_ACCOSTTOGOFITNESSFUNC_H_
#define FLEX_NEURALNET_ACCOSTTOGOFITNESSFUNC_H_

#include "flexnnet.h"

namespace flexnnet
{
   template<class _OutTyp>
   class ActorCriticCostToGoFitnessFunc : public BaseFitnessFunc
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
      calc_td_error_gradient(const _OutTyp& _externR, const _OutTyp& _V_est, const _OutTyp _prev_V_est);

      const ValarrMap&
      calc_actor_error_gradient(const _OutTyp& _target, const _OutTyp& _nnout);

   private:
      double gamma{DEFAULT_GAMMA};
      double sse, series_sse;
      size_t sample_count, series_len;
      ValarrMap td_gradient;
   };

   template<class OutTyp>
   void
   ActorCriticCostToGoFitnessFunc<OutTyp>::set_gamma(double _gamma)
   {
      if (_gamma < 0 || _gamma > 1.0)
      {
         std::ostringstream err_str;
         err_str
            << "Error : ActorCriticCostToGoFitnessFunc.set_gamma() - illegal vectorize" << _gamma << "\n";
         throw std::invalid_argument(err_str.str());
      }

      gamma = _gamma;
   }

   template<class OutTyp>
   void
   ActorCriticCostToGoFitnessFunc<OutTyp>::clear(void)
   {
      sse = 0;
      sample_count = 0;
      series_sse = 0;
      series_len = 0;
   }

   template<class OutTyp>
   void
   ActorCriticCostToGoFitnessFunc<OutTyp>::new_series(void)
   {
      sse += series_sse * series_sse / (series_len+1);
      sample_count++;

      series_len = 0;
      series_sse = 0;
   }

   template<class OutTyp>
   const ValarrMap&
   ActorCriticCostToGoFitnessFunc<OutTyp>::calc_actor_error_gradient(const OutTyp& _tgt, const OutTyp& _nnout)
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
      ActorCriticCostToGoFitnessFunc::series_sse = sse + ActorCriticCostToGoFitnessFunc::series_sse;
      series_len++;

      return td_gradient;
   }

   template<class OutTyp>
   inline
   const ValarrMap&
   ActorCriticCostToGoFitnessFunc<OutTyp>::calc_td_error_gradient(const OutTyp& _externR, const OutTyp& _V_est, const OutTyp _prev_V_est)
   {
      const ValarrMap& externR_vamap = _externR.value_map();
      const ValarrMap& V_est_vamap = _V_est.value_map();
      const ValarrMap& prev_V_est_vamap = _prev_V_est.value_map();
      std::cout << "size of _externR " << _externR.value_map().size() << "\n"; //<< " " << _V_est.at(id)[0] << "\n";
      std::cout << "size of externR_vamap " << externR_vamap.size() << "\n"; //<< " " << _V_est.at(id)[0] << "\n";
      std::cout << "size of externR_vamap.begin()->second.size() " << externR_vamap.begin()->second.size() << "\n"; //<< " " << _V_est.at(id)[0] << "\n";

      double sse = 0;
      int i = 0;
      for (const auto& externR : externR_vamap)
      {
         const std::string id = externR.first;
         td_gradient[id] = externR.second + gamma * _V_est.at(id) - _prev_V_est.at(id);

         std::cout << "size of R " << externR.second.size() << "\n"; //<< " " << _V_est.at(id)[0] << "\n";
         std::cout << "episode err " << i << " " << td_gradient[id].sum() << "\n";
         i++;

         sse += td_gradient[id].sum();
      }
      ActorCriticCostToGoFitnessFunc::series_sse = sse + ActorCriticCostToGoFitnessFunc::series_sse;
      series_len++;

      return td_gradient;
   }

   template<class _OutTyp>
   double
   ActorCriticCostToGoFitnessFunc<_OutTyp>::calc_fitness()
   {
      std::cout << "calc fitness " << sample_count << " " << sse << "\n";

      return (sample_count > 0) ? (sse / sample_count) : 0;
   }
}

#endif //FLEX_NEURALNET_TDCOSTTOGOFITNESSFUNC_H_
