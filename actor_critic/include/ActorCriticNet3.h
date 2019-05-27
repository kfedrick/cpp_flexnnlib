/*
 * ActorCriticNet3.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ACTORCRITIC_NET3_H_
#define FLEX_NEURALNET_ACTORCRITIC_NET3_H_

#include <vector>

#include "ActorCriticOutput.h"
#include "Action.h"
#include "ActorNet3.h"
#include "AdaptiveCriticNet3.h"

using namespace std;

namespace flexnnet
{

   class ActorCriticNet3
   {
   public:
      ActorCriticNet3 (ActorNet3 *_actor, AdaptiveCriticNet3 *_adaptCritic);
      virtual ~ActorCriticNet3 ();

      void set_stochastic_action (bool _val);
      void set_print_gradient (bool _val);

      ActorNet3 *get_actor ();
      AdaptiveCriticNet3 *get_adaptive_critic ();
      bool get_stochastic_action ();

      const ActorCriticOutput &operator() (const Pattern &ipattern, unsigned int recurStep = 1);
      const ActorCriticOutput &
      operator() (const Pattern &ipattern, const Pattern &apattern, unsigned int recurStep = 1);

      void clear_error (unsigned int timeStep = 1);
      void backprop (const vector<double> &_eVec, unsigned int timeStep = 1);

   private:
      ActorNet3 *actor;
      AdaptiveCriticNet3 *adaptive_critic;

      ActorCriticOutput last_actor_critic_output;

      bool stochastic_action;
      bool print_gradient;
   };

   inline
   void ActorCriticNet3::set_stochastic_action (bool _val)
   {
      stochastic_action = _val;
   }

   inline
   bool ActorCriticNet3::get_stochastic_action ()
   {
      return stochastic_action;
   }

   inline
   void ActorCriticNet3::set_print_gradient (bool _val)
   {
      print_gradient = _val;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_ACTORCRITIC_NET3_H_ */
