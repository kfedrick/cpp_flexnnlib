//
// Created by kfedrick on 5/17/21.
//

#ifndef FLEX_NEURALNET_ENVIRONMENT_H_
#define FLEX_NEURALNET_ENVIRONMENT_H_

#include <EnvironReinforcement.h>

namespace flexnnet
{
   /**
    * Environment specifies the interface for the basis environment simulator.
    *
    * @tparam Features - The class encoding the state features (implements NetworkInput)
    * @tparam Action - the class that encodes the actions (implements ActionView)
    * @tparam RSZ - the number of reinforcement signals returned by the environment (>0)
    */
   template<class Features, typename Action, size_t RSZ>
   class Environment
   {
   public:
      static const size_t R_SIGNAL_COUNT = RSZ;

   public:
      /**
       * Reset to the start of the episode and return the start state.
       *
       * @return - The start state
       */
      virtual const Features& reset(void) = 0;

      /**
       * Advance the episode to the next state given the specified dynamic
       * feedback; return the next state.
       *
       * @param _dynamic_input
       * @return  - the next state
       */
      const Features& next(const Action& _action);

      /**
       * Return the current state vector.
       * @return
       */
      virtual const Features& state() const = 0;

      /**
       * Return the reinforcement signal for the current state.
       *
       * @return
       */
      virtual const EnvironReinforcement<RSZ>& get_reinforcement() const = 0;

      /**
       * Return a boolean indicating whether the episode has reached a terminal state.
       *
       * @return - boolean indication that the episode is in a terminal state.
       */
      virtual bool is_terminal(void) const = 0;
   };
} // end namespace flexnnet

#endif //FLEX_NEURALNET_ENVIRONMENT_H_
