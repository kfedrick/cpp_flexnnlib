//
// Created by kfedrick on 3/3/21.
//

#ifndef FLEX_NEURALNET_ACTORCRITICEVALUATOR_H_
#define FLEX_NEURALNET_ACTORCRITICEVALUATOR_H_

#include <iostream>
#include <flexnnet.h>
#include <Exemplar.h>
#include <ExemplarSeries.h>
#include <BasicEvalConfig.h>
#include "Reinforcement.h"

namespace flexnnet
{
   template<class State, class Action, size_t N,
      template<class, class, size_t> class NN,
      template<class, class, size_t> class Env,
      template<class> class FitnessFunc>
   class ActorCriticEvaluator : public BasicEvalConfig, public FitnessFunc<Reinforcement<N>>
   {
      using NNTyp = NN<State, Action, N>;
      using EnvTyp = Env<State, Action, N>;

   public:
      ActorCriticEvaluator();

      std::tuple<double, double>
      evaluate(NNTyp& _nnet, EnvTyp& _env);

   protected:
      double
      evaluate_subsampling(size_t _s_index, NNTyp& _nnet, EnvTyp& _env);

      void
      evaluate_episode(size_t _s_index, NNTyp& _nnet, EnvTyp& _env);
   };

   template<class State, class Action, size_t N, template<class, class, size_t> class NN,
      template<class, class, size_t> class Env,
      template<class> class FitnessFunc>
   ActorCriticEvaluator<State,
               Action,
               N,
               NN,
               Env,
               FitnessFunc>::ActorCriticEvaluator() : FitnessFunc<Reinforcement<N>>()
   {
   }


   template<class State, class Action, size_t N, template<class, class, size_t> class NN,
      template<class, class, size_t> class Env,
      template<class> class FitnessFunc>
   std::tuple<double, double>
   ActorCriticEvaluator<State,
               Action,
               N,
               NN,
               Env,
               FitnessFunc>::evaluate(NNTyp& _nnet, EnvTyp& _env)
   {
      //std::cout << "ActorCriticEvaluator.evaluate()\n";

      // Vector to hold performance results for each sampling
      size_t scount = sampling_count();
      std::valarray<double> perf(scount);
      FitnessFunc<Reinforcement<N>>::clear();

      // Iterate through all exemplars in the training set_weights
      for (size_t sample_ndx = 0; sample_ndx < scount; sample_ndx++)
      {
         perf[sample_ndx] = evaluate_subsampling(sample_ndx, _nnet, _env);
      }

      // Calculate performance mean and standard deviation across samples
      double sample_mean = perf.sum() / scount;
      double stdev = 0;
      for (size_t i = 0; i < scount; i++)
         stdev += (sample_mean - perf[i]) * (sample_mean - perf[i]);

      double svar = (scount > 30) ? stdev / (scount - 1) : stdev / scount;
      stdev = sqrt(svar);

      //std::cout << "eval stats [" << scount << "]: (" << sample_mean << "," << stdev << ")\n";

      return std::tuple<double,double>(sample_mean,stdev);
   }

   template<class State, class Action, size_t N, template<class, class, size_t> class NN,
      template<class, class, size_t> class Env,
      template<class> class FitnessFunc>
   double
   ActorCriticEvaluator<State,
               Action,
               N,
               NN,
               Env,
               FitnessFunc>::evaluate_subsampling(size_t _sample_ndx, NNTyp& _nnet, EnvTyp& _env)
   {
      FitnessFunc<Reinforcement<N>>::clear();

      for (int sample_no=0; sample_no<1; sample_no++)
      {
         evaluate_episode(sample_no, _nnet, _env);
         FitnessFunc<Reinforcement<N>>::new_series();
      }

      double err = FitnessFunc<Reinforcement<N>>::calc_fitness();
      //std::cout << "fitness calc_fitness " << err << "\n";
      return err;
   }

   template<class State, class Action, size_t N, template<class, class, size_t> class NN,
      template<class, class, size_t> class Env,
      template<class> class FitnessFunc>
   void
   ActorCriticEvaluator<State,
               Action,
               N,
               NN,
               Env,
               FitnessFunc>::evaluate_episode(size_t _s_index, NNTyp& _nnet, EnvTyp& _env)
   {
      Reinforcement<N> V_est;
      Reinforcement<N> previous_V_est;
      Reinforcement<N> zero_V_est("R");
      zero_V_est.decode({{0.0}});

      const State& state = _env.clear_learning_rate_adjustments();
      const std::tuple<Action, Reinforcement<N>>& nnout = _nnet.activate(state);
      V_est = std::get<1>(nnout);

      _env.next(std::get<0>(nnout).get_action());

      while (!_env.is_terminal())
      {
         // V(t)
         const Reinforcement<N>& R = _env.get_reinforcement();
         //std::cout << " --- sizeof R : " << R.value_map().begin()->second.size() << "\n" << std::flush;
         //std::cout << "R name " << R.value_map().begin()->first << "\n";

         // V estimate (t)
         const State& state = _env.state();
         const std::tuple<Action, Reinforcement<N>>& nnout = _nnet.activate(state);
         previous_V_est = V_est;
         V_est = std::get<1>(nnout);

         //std::cout << "V_est : " << V_est[0] << "\n" << std::flush;
         //std::cout << " ----- sizeof R is now : " << R.value_map().begin()->second.size() << "\n" << std::flush;
         //std::cout << "AC Eval V_est[0].name() " << V_est.value_map().begin()->first << "\n";
         //std::cout << "AC Eval get<1>[0].name() " << std::get<1>(nnout).value_map().begin()->first << "\n";

         //const ValarrMap
         //   & td_gradient = FitnessFunc<Reinforcement<N>>::calc_td_error_gradient(R, V_est, previous_V_est);
         const ValarrMap
            & td_gradient = FitnessFunc<Reinforcement<N>>::calc_td_error_gradient(zero_V_est, V_est, previous_V_est);
         //std::cout << "after calc td error gradient\n" << std::flush;

         _env.next(std::get<0>(nnout).get_action());
      }

      const Reinforcement<N>& R = _env.get_reinforcement();
      //std::cout << "final (R,Vest) = " << R[0] << "," << V_est[0] << "\n" << std::flush;
      const ValarrMap& td_gradient = FitnessFunc<Reinforcement<N>>::calc_td_error_gradient(R, zero_V_est, V_est);
   }
}

#endif //FLEX_NEURALNET_SERIESEVALUATOR_H_
