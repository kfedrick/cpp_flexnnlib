/*
 * GreedyActorCriticNet.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_GREEDYACTORCRITIC_NET_H_
#define FLEX_NEURALNET_GREEDYACTORCRITIC_NET_H_

#include <vector>

#include "ActorCriticOutput.h"
#include "Action.h"
#include "GreedyActorNet.h"
#include "AdaptiveCriticNet.h"

using namespace std;

namespace flexnnet
{

   class GreedyActorCriticNet
   {
   public:
      GreedyActorCriticNet (const Pattern &stateVec, const set<string> &_actionSet, AdaptiveCriticNet *_adaptCritic);
      virtual ~GreedyActorCriticNet ();

      void set_stochastic_action (bool _val);

      ActorNet *get_actor ();
      AdaptiveCriticNet *get_adaptive_critic ();

      const ActorCriticOutput &operator() (const Pattern &ipattern, unsigned int recurStep = 1);

      void clear_error (unsigned int timeStep = 1);
      void backprop (const vector<double> &_eVec, unsigned int timeStep = 1);

   private:
      GreedyActorNet *actor;
      AdaptiveCriticNet *adaptive_critic;

      ActorCriticOutput last_actor_critic_output;

      bool stochastic_action;
   };

   inline
   void GreedyActorCriticNet::set_stochastic_action (bool _val)
   {
      stochastic_action = _val;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_GREEDYACTORCRITIC_NET_H_ */
