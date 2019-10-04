//
// Created by kfedrick on 9/15/19.
//

#ifndef FLEX_NEURALNET_EPISODE_H_
#define FLEX_NEURALNET_EPISODE_H_

#include <utility>

namespace flexnnet
{
   template<class _InType, class _OutType>
   class Episode
   {
      /**
       * Reset to the start of the episode and return the start state.
       *
       * @return - The start state
       */
       _InType& reset(void);

      /**
       * Advance the episode to the next state given the specified dynamic feedback; return
       * the next state.
       *
       * @param _dynamic_input
       * @return  - the next state
       */
      _InType& next(_OutType& _dynamic_feedback);

      /**
       * Return the expected target or reinforcement signal for current state.
       *
       * @return
       */
      _OutType& target(void);

      /**
       * Return the reinforcement signal for the current state.
       *
       * @return
       */
      std::pair<bool, double>& get_reinforcement(void);

      /**
       * Return a boolean indicating whether the episode has reached a terminal state.
       *
       * @return - boolean indication that the episode is in a terminal state.
       */
      bool is_terminal(void);

   };
}

#endif //FLEX_NEURALNET_EPISODE_H_
