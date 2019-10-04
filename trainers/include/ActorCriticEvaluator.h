//
// Created by kfedrick on 9/29/19.
//

#ifndef FLEX_NEURALNET_ACTORCRITICEVALUATOR_H_
#define FLEX_NEURALNET_ACTORCRITICEVALUATOR_H_

#include <cstddef>
#include <iostream>
#include <valarray>

#include "TDTrainerConfig.h"
#include "TDEvaluatorConfig.h"

namespace flexnnet
{
   template<class _State, class _Action,
      template<class, class> class _Env, template<class __S, class __A> class _Agent,
      template<class> class _ErrorFunc, TDForecastMode _MODE>
   class ActorCriticEvaluator : public TDEvaluatorConfig<_MODE>
   {
      /**
       * Struct to return results from this trainer.
       */
      struct Results
      {
         // The critic error - how well the critic estimated the actual
         // reinforcement signals from the environment.
         //
         double critic_error;

         // A measure of how well the actor-critic performed in maximizing positive
         // reinforcement on the problem.
         //
         double actor_critic_perf;
      };

   protected:

      struct AgentOutput
      {
         const _Action& action;
         double est_rsig;
      };

   public:
      /**
       * Evaluate the performance of the agent over the specified number of samplings
       * episodes.
       *
       * @param _nnet
       * @param _tstset
       * @return
       */
      Results evaluate(_Agent<_State, _Action>& _agent, _Env<_State, _Action>& _env);

   protected:
      /**
       * Evaluate the performance of the agent on one episode using TD final
       * reinforcement mode.
       *
       * @param _agent
       * @param _env
       * @return
       */
      Results evaluate_episode(_Agent<_State, _Action>& _agent, _Env<_State, _Action>& _env);

    };

   template<class _State, class _Action,
      template<class, class> class _Env, template<class, class> class _Agent,
      template<class> class _ErrorFunc, TDForecastMode _MODE>
   typename ActorCriticEvaluator<_State, _Action, _Env, _Agent, _ErrorFunc, _MODE>::Results
   ActorCriticEvaluator<_State, _Action, _Env, _Agent, _ErrorFunc, _MODE>::evaluate(_Agent<_State,
                                                                                           _Action>& _agent, _Env<
      _State,
      _Action>& _env)
   {
      std::cout << "ActorCriticEvaluator::evaluate()\n";

      typename ActorCriticEvaluator<_State, _Action, _Env, _Agent, _ErrorFunc, _MODE>::Results results;

      std::valarray<double> critic_errors;
      std::valarray<double> scores;

      // Vector to hold performance results for each sampling
      size_t scount = TDEvaluatorConfig<_MODE>::sampling_count();

      // Iterate through all exemplars in the training set
      for (size_t sampling_ndx = 0; sampling_ndx < scount; sampling_ndx++)
      {
         results = evaluate_episode(_agent, _env);

         critic_errors[sampling_ndx] = results.critic_error;
         scores[sampling_ndx] = results.actor_critic_perf;
      }

      // TODO - calculate statistics across samplings

      return results;
   }

   template<class _State, class _Action,
      template<class, class> class _Env, template<class, class> class _Agent,
      template<class> class _ErrorFunc, TDForecastMode _MODE>
   typename ActorCriticEvaluator<_State, _Action, _Env, _Agent, _ErrorFunc, _MODE>::Results
   ActorCriticEvaluator<_State, _Action, _Env, _Agent, _ErrorFunc, _MODE>::evaluate_episode(_Agent<_State,
                                                                                                   _Action>& _agent, _Env<
      _State,
      _Action>& _env)
   {
      double est_rsig, prev_est_rsig, tgt_rsig;
      double gamma = TDEvaluatorConfig<_MODE>::td_discount();
      double td_error;
      double mse = 0;

      std::pair<bool, double> extern_rsig;

      typename ActorCriticEvaluator<_State, _Action, _Env, _Agent, _ErrorFunc, _MODE>::Results results;
      typename ActorCriticEvaluator<_State, _Action, _Env, _Agent, _ErrorFunc, _MODE>::AgentOutput agent_rval;

      // Reset environment to start state and get activate actor-critic
      // agent to determine initial action.
      //
      _State state = _env.reset();
      agent_rval = _agent.activate(state);

      // Iterate through remaining steps of episode until we reach a terminal state.
      //
      size_t episode_len = 1;
      do
      {
         // Update the environment state based on the actor-critic action
         // and get the external reinforcement signal.
         //
         state = _env.next(agent_rval.action);
         extern_rsig = _env.reinforcement();

         // Save current critic estimate for the external reinforcement signal
         // and activate the agent for the current state to get a new action
         // and estimate.
         //
         est_rsig = agent_rval.est_rsig;
         agent_rval = _agent.activate(state);

         // If we receive external reinforcement only at the end of the episode
         // use the final reinforcement td formula
         if (TDEvaluatorConfig<_MODE>::td_forecast_mode == FINAL_COST)
         {
            // It there is an external reinforcement signal, use it as the
            // training target for the reinforcement signal; otherwise use
            // the next estimate.
            //
            tgt_rsig = (extern_rsig.first) ? extern_rsig.second : agent_rval.est_rsig;

            // Calulate the temporal difference error
            td_error = tgt_rsig - est_rsig;
         }

         // Else use COST_TO_GO temporal difference error
         else
         {
            if (_env.is_terminal())
               td_error = extern_rsig.second - prev_est_rsig;
            else
               td_error = extern_rsig.second + gamma * agent_rval.est_rsig - prev_est_rsig;
         }

         // Accumulate squared temporal difference error.
         mse += td_error * td_error;

         episode_len++;
      }
      while (!_env.is_terminal());

      // Calculate sample mean squared error
      mse /= (episode_len - 1);

      // Pack return structure.
      results.critic_error = mse;

      return results;
   }


}

#endif //FLEX_NEURALNET_ACTORCRITICEVALUATOR_H_
