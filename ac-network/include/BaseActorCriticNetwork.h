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
      typedef FeatureSetImpl<StateActionTuple> StateAction;

   public:
      BaseActorCriticNetwork(
         const NeuralNet<State, Action>& _actor, const NeuralNet<StateAction, Reinforcement<N>>& _critic);

      BaseActorCriticNetwork(BaseActorCriticNetwork& _acnnet);

      ~BaseActorCriticNetwork();

      BaseActorCriticNetwork& operator=(const BaseActorCriticNetwork& _acnnet);

      /**
       * Transact a single episode on the environment
       * @param _env
       */
      const std::tuple<Action, Reinforcement<N>>& activate(const State& _state);

      const void backprop_actor(const ValarrMap& _err);
      const void backprop_critic(const ValarrMap& _ones);

      const std::tuple<Action, Reinforcement<N>>& value(void) const;

      NeuralNet<State, Action>& get_actor();
      NeuralNet<StateAction, Reinforcement<N>>& get_critic();

   private:
      NeuralNet<State, Action> actor;
      NeuralNet<StateAction, Reinforcement<N>> critic;

      StateAction critic_input;

      Action action;
      Reinforcement<N> R;
      std::tuple<Action, Reinforcement<N>> nn_output;
   };

   template<typename S, typename A, size_t N>
   BaseActorCriticNetwork<S,A,N>::BaseActorCriticNetwork(
      const NeuralNet<S, A>& _actor, const NeuralNet<StateAction, Reinforcement<N>>& _critic)
      : actor(_actor), critic(_critic)
   {
      critic_input = StateAction({"F0", "F1"});
      R = Reinforcement<N>(_critic.get_layers().begin()->first);
      //std::cout << "ACNet critic R name " << R.get_feature_names()[0] << "\n";

   }

   template<typename S, typename A, size_t N>
   BaseActorCriticNetwork<S,A,N>::BaseActorCriticNetwork(BaseActorCriticNetwork& _acnnet): actor(_acnnet.get_actor()), critic(_acnnet.get_critic())
   //BaseActorCriticNetwork<S,A,N>::BaseActorCriticNetwork(BaseActorCriticNetwork& _acnnet): actor(NeuralNet<S, A>()), critic(NeuralNet<StateAction, Reinforcement<N>>())
   {
      R = _acnnet.R;
      critic_input = _acnnet.critic_input;
      nn_output = _acnnet.nn_output;
      action = _acnnet.action;

      //std::cout << "------------- Copy Constructor ACNet critic R name " << R.get_feature_names()[0] << "\n";

   }

   template<typename S, typename A, size_t N>
   BaseActorCriticNetwork<S,A,N>& BaseActorCriticNetwork<S,A,N>::operator=(const BaseActorCriticNetwork& _acnnet)
   {
      R = _acnnet.R;
      critic_input = _acnnet.critic_input;
      nn_output = _acnnet.nn_output;
      action = _acnnet.action;
      //std::cout << "ACNet operator= critic R name " << R.get_feature_names()[0] << "\n";

      return *this;
   }

   template<typename S, typename A, size_t N>
   BaseActorCriticNetwork<S,A,N>::~BaseActorCriticNetwork()
   {
   }

   template<typename S, typename A, size_t N>
   const std::tuple<A, Reinforcement<N>>& BaseActorCriticNetwork<S,A,N>::activate(const S& _state)
   {
      //std::cout << "activate ACNet critic R name " << R.get_feature_names()[0] << "\n";

      // Activate the actor network and save it's output
      action = actor.activate(_state);


      // replace actor action with random
      double r = ((double) rand()) / RAND_MAX - 0.5;
      std::get<0>(action.get_features()).decode({r});

      // Marshal inputs for critic
      critic_input.get_features() = std::tuple_cat(_state.get_features(), action.get_features());

/*      std::valarray<double> v = std::get<0>(_state.get_features()).get_encoding();
      for (int i=0; i<v.size(); i++)
      {
         std::cout << v[i] << " ";
      }
      std::cout << "\n";*/

/*      std::cout << "action input to critic: ";
      std::valarray<double> v = std::get<0>(action.get_features()).get_encoding();
      for (int i=0; i<v.size(); i++)
      {
         std::cout << v[i] << " ";
      }
      std::cout << "\n";*/

      // Activate critic
      R = critic.activate(critic_input);

      // Return the AC network output (the action and the value function estimate)
      std::tuple<A, Reinforcement<N>> ret(action,R);

      nn_output = ret;
      return nn_output;
   }

   template<typename S, typename A, size_t N>
   const void BaseActorCriticNetwork<S,A,N>::backprop_actor(const ValarrMap& _err)
   {
      critic.backprop(_err);

      const ValarrMap& critic_dEdx = critic.get_dEdx();
      actor.backprop(critic_dEdx);
   }

   template<typename S, typename A, size_t N>
   const void BaseActorCriticNetwork<S,A,N>::backprop_critic(const ValarrMap& _ones)
   {
      //std::cout << "critic.backprop(" << _ones.begin()->first << ") " << _ones.begin()->second[0] << "\n";
      critic.backprop(_ones);
   }

   template<typename S, typename A, size_t N>
   const std::tuple<A, Reinforcement<N>>& BaseActorCriticNetwork<S,A,N>::value(void) const
   {
      return nn_output;
   }

   template<typename S, typename A, size_t N>
   NeuralNet<S, A>& BaseActorCriticNetwork<S,A,N>::get_actor()
   {
      return actor;
   }

   template<typename S, typename A, size_t N>
   NeuralNet<typename BaseActorCriticNetwork<S,A,N>::StateAction, Reinforcement<N>>& BaseActorCriticNetwork<S,A,N>::get_critic()
   {
      return critic;
   }

}  // end namespace flexnnet

#endif //FLEX_NEURALNET_BASEACTORCRITICNETWORK_H_
