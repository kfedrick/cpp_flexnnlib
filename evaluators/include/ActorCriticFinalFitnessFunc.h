//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_ACFINALFITNESSFUNC_H_
#define FLEX_NEURALNET_ACFINALFITNESSFUNC_H_

#include "flexnnet.h"

namespace flexnnet
{
   template<class OutTyp>
   class ActorCriticFinalFitnessFunc : public BaseFitnessFunc
   {
   public:

      void
      clear(void);

      void
      new_series(void);

      double
      calc_fitness(void);

      const ValarrMap&
      calc_actor_error_gradient(const OutTyp& _target, const OutTyp& _nnout);

      const ValarrMap&
      calc_td_error_gradient(const OutTyp& _target, const OutTyp& _V_est, const OutTyp _prev_V_est);

   private:
      double sse, series_sse;
      size_t sample_count, series_len;
      ValarrMap td_gradient;
   };

   template<class OutTyp>
   inline
   void
   ActorCriticFinalFitnessFunc<OutTyp>::clear(void)
   {
      sse = 0;
      series_sse = 0;
      sample_count = 0;
      series_len = 0;
   }

   template<class OutTyp>
   inline
   void
   ActorCriticFinalFitnessFunc<OutTyp>::new_series(void)
   {
      //std::cout << "--- new_series " << series_len << " " << series_sse << "\n";
      //std::cout << "------ sse " << sse << "\n";

      if (series_len > 0)
      {
         sse += series_sse * series_sse / series_len;
         sample_count++;
      }

      series_len = 0;
      series_sse = 0;
   }

   template<class OutTyp>
   const ValarrMap&
   ActorCriticFinalFitnessFunc<OutTyp>::calc_actor_error_gradient(const OutTyp& _tgt, const OutTyp& _nnout)
   {
      std::valarray<double> sqrdiff;

      const ValarrMap& tgt_vamap = _tgt.value_map();
      const ValarrMap& nnout_vamap = _nnout.value_map();
      //std::cout << "FINAL size of tgt_vamap.begin()->second.size() " << tgt_vamap.begin()->second.size() << "\n"; //<< " " << _V_est.at(id)[0] << "\n";
      //std::cout << "FINAL size of nnout_vamap name " << nnout_vamap.begin()->first << "\n"; //<< " " << _V_est.at(id)[0] << "\n";

      double sse = 0;
      for (const auto& tgt : tgt_vamap)
      {
         const std::string id = tgt.first;
         //std::cout << "tgt.first " << tgt.first << "\n";


         td_gradient[id] = tgt.second - nnout_vamap.at(id);
         sse += td_gradient[id].sum();
      }
      ActorCriticFinalFitnessFunc::series_sse += sse;
      series_len++;

      return td_gradient;
   }

   template<class OutTyp>
   inline
   const ValarrMap&
   ActorCriticFinalFitnessFunc<OutTyp>::calc_td_error_gradient(const OutTyp& _externR, const OutTyp& _V_est, const OutTyp _prev_V_est)
   {
      std::valarray<double> sqrdiff;

      const ValarrMap& externR_vamap = _externR.value_map();
      const ValarrMap& V_est_vamap = _V_est.value_map();
      const ValarrMap& prev_V_est_vamap = _prev_V_est.value_map();
      //std::cout << "FINAL size of externR_vamap.begin()->second.size() " << externR_vamap.begin()->second.size() << "\n"; //<< " " << _V_est.at(id)[0] << "\n";
      //std::cout << "size of externR_vamap " << externR_vamap.size() << "\n";

      double sse = 0;
      int i = 0;
      for (const auto& externR : externR_vamap)
      {
         const std::string id = externR.first;
         //std::cout << "externR.first " << externR.first << "\n";
         //std::cout << "V_est[0].name() " << V_est_vamap.begin()->first << "\n";

         if (externR.second[0] != 0)
         {
            //if (externR.second.size() > 0)
            td_gradient[id] = externR.second - prev_V_est_vamap.at(id);
            //td_gradient[id] = externR.second - V_est_vamap.at(id);

            //std::cout << "episode err " << i << " R-V = " << td_gradient[id].sum() << "\n";
         }
         else
         {
            td_gradient[id] = V_est_vamap.at(id) - prev_V_est_vamap.at(id);

            //std::cout << "episode err " << i << " V(t)-V(t-1)=" << td_gradient[id].sum() << "\n";
         }

         sse += td_gradient[id].sum();
         i++;
      }
      ActorCriticFinalFitnessFunc::series_sse += sse;
      series_len++;
      //std::cout << "series_sse " << i << ActorCriticFinalFitnessFunc::series_sse << "\n";

      return td_gradient;
   }

   template<class OutTyp>
   inline
   double
   ActorCriticFinalFitnessFunc<OutTyp>::calc_fitness()
   {
      //std::cout << "--- calc fitness " << sample_count << " " << sse << "\n";

      return (sample_count > 0) ? (sqrt(sse) / sample_count) : 0;
   }
}

#endif //FLEX_NEURALNET_TDFINALFITNESSFUNC_H_
