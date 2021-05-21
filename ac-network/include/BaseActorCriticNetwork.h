//
// Created by kfedrick on 5/11/21.
//

#ifndef FLEX_NEURALNET_BASEACTORCRITICNETWORK_H_
#define FLEX_NEURALNET_BASEACTORCRITICNETWORK_H_

#include <tuple>
#include <BaseNeuralNet.h>

namespace flexnnet
{
   template<typename Action, typename R, typename ...Features>
   class BaseActorCriticNetwork
   {
   public:
      /**
       * Transact a single episode on the environment
       * @param _env
       */
      const std::tuple<R, Action>& activate(const Features&... _features);

   protected:
      void unpack_extern_inputs(const Features&... _features);

   private:
      template<typename F, typename ...Fr>
      void unpack_extern_inputs_impl(const F _feature, const Fr&... _features);
      void unpack_extern_inputs_impl() {};


   private:
      BaseNeuralNet actor;
      BaseNeuralNet critic;

      std::tuple<R, Action> nn_output;
      std::tuple<Features...> raw_extern_inputs;
      std::vector<std::valarray<double>> vectorized_extern_inputs;
   };

   template<typename Action, typename R, typename ...Fs>
   const std::tuple<R, Action>& BaseActorCriticNetwork<Action,R,Fs...>::activate(const Fs&... _features)
   {
      unpack_extern_inputs(_features...);
   }

   template<typename Action, typename R, typename ...Fs>
   void BaseActorCriticNetwork<Action,R,Fs...>::unpack_extern_inputs(const Fs&... _features)
   {
      unpack_extern_inputs_impl(_features...);
   }

   template<typename Action, typename R, typename ...Fs>
   template<typename F, typename ...Fr>
   void BaseActorCriticNetwork<Action,R,Fs...>::unpack_extern_inputs_impl(const F _first, const Fr&... _remainder)
   {
      std::cout << _first.name() << "\n";
      unpack_extern_inputs_impl(_remainder...);
   }

}  // end namespace flexnnet

#endif //FLEX_NEURALNET_BASEACTORCRITICNETWORK_H_
