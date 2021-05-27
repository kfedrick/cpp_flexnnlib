//
// Created by kfedrick on 5/11/21.
//

#ifndef FLEX_NEURALNET_BASEACTORCRITICNETWORK_H_
#define FLEX_NEURALNET_BASEACTORCRITICNETWORK_H_

#include <tuple>
#include <BaseNeuralNet.h>
#include <NeuralNet.h>
#include <Reinforcement.h>
#include <RawFeatureSet.h>

namespace flexnnet
{
   template<typename State, typename Action, size_t N>
   class BaseActorCriticNetwork
   {
   private:
      typedef decltype(std::tuple_cat(std::declval<State>().get_features(),
                                                   std::declval<Action>().get_features())) StateActionTuple;
      typedef FeatureSet<StateActionTuple> StateAction;

   public:
      BaseActorCriticNetwork(
         const NeuralNet<State, Action>& _actor, const NeuralNet<StateAction, Reinforcement<N>>& _critic);
      ~BaseActorCriticNetwork();

      /**
       * Transact a single episode on the environment
       * @param _env
       */
      const std::tuple<Action, Reinforcement<N>>& activate(const State& _state);
      const void backprop(const std::valarray<double>& _r);

      const std::tuple<Action, Reinforcement<N>>& value(void) const;

      const NeuralNet<State, Action>& get_actor() const;
      const NeuralNet<StateAction, Reinforcement<N>>& get_critic() const;

   private:
      NeuralNet<State, Action> actor;
      NeuralNet<StateAction, Reinforcement<N>> critic;

      StateAction critic_input;
      std::tuple<Action, Reinforcement<N>> nn_output;
   };

   template<typename S, typename A, size_t N>
   BaseActorCriticNetwork<S,A,N>::BaseActorCriticNetwork(
      const NeuralNet<S, A>& _actor, const NeuralNet<StateAction, Reinforcement<N>>& _critic)
      : actor(_actor), critic(_critic)
   {
      critic_input = StateAction({"F0", "action"});
   }

   template<typename S, typename A, size_t N>
   BaseActorCriticNetwork<S,A,N>::~BaseActorCriticNetwork()
   {
   }

   template<typename S, typename A, size_t N>
   const std::tuple<A, Reinforcement<N>>& BaseActorCriticNetwork<S,A,N>::activate(const S& _state)
   {

      // Activate the actor network and save it's output
      std::get<0>(nn_output) = actor.activate(_state);
      std::get<0>(std::get<0>(nn_output).get_features()).decode({1});


      std::cout << "state " << std::get<0>(_state.get_features()).get_encoding()[1] << "\n";
      std::cout << "action " << std::get<0>(std::get<0>(nn_output).get_features()).get_encoding()[0] << "\n";


      std::cout << "marshal_critic_input\n";

      const A& action = std::get<0>(nn_output);
      critic_input.get_features() = std::tuple_cat(_state.get_features(), action.get_features());

      std::cout << "activate critic\n";
      std::get<1>(nn_output) = critic.activate(critic_input);
      std::cout << "after activate critic\n" << std::flush;

      std::cout << "action is still " << std::get<0>(std::get<0>(nn_output).get_features()).get_encoding()[0] << "\n";

      // Return the AC network output (the action and the value function estimate)
      return nn_output;
   }

   template<typename S, typename A, size_t N>
   const void BaseActorCriticNetwork<S,A,N>::backprop(const std::valarray<double>& _r)
   {

   }

   template<typename S, typename A, size_t N>
   const std::tuple<A, Reinforcement<N>>& BaseActorCriticNetwork<S,A,N>::value(void) const
   {
      return nn_output;
   }

   template<typename S, typename A, size_t N>
   const NeuralNet<S, A>& BaseActorCriticNetwork<S,A,N>::get_actor() const
   {
      return actor;
   }

   template<typename S, typename A, size_t N>
   const NeuralNet<typename BaseActorCriticNetwork<S,A,N>::StateAction, Reinforcement<N>>& BaseActorCriticNetwork<S,A,N>::get_critic() const
   {
      return critic;
   }

}  // end namespace flexnnet

#endif //FLEX_NEURALNET_BASEACTORCRITICNETWORK_H_
