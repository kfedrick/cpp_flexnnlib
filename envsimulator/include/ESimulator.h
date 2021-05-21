//
// Created by kfedrick on 9/15/19.
//

#ifndef FLEX_NEURALNET_ESIMULATOR_H_
#define FLEX_NEURALNET_ESIMULATOR_H_

#include <string>
#include <map>

#include <flexnnet.h>
#include <FeatureVector.h>

namespace flexnnet
{
   class ESimulator
   {
   public:
      /**
       * Return the current state vector.
       * @return
       */
      virtual const FeatureVector& state() = 0;

      /**
       * Reset to the start of the episode and return the start state.
       *
       * @return - The start state
       */
      virtual const FeatureVector& reset(void) = 0;

      /**
       * Advance the episode to the next state given the specified dynamic
       * feedback; return the next state.
       *
       * @param _dynamic_input
       * @return  - the next state
       */
      template<class ActionTyp>
      const FeatureVector& next(const ActionTyp& _action);

      /**
       * Return the reinforcement signal for the current state.
       *
       * @return
       */
      virtual const std::map<std::string, double>& get_reinforcement() const = 0;

      /**
       * Return a boolean indicating whether the episode has reached a terminal state.
       *
       * @return - boolean indication that the episode is in a terminal state.
       */
      virtual bool is_terminal(void) const = 0;
   };
}

#endif //FLEX_NEURALNET_ESIMULATOR_H_
