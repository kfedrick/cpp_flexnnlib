/*
 * PBActorCriticNet.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_PBACTORCRITIC_NET_H_
#define FLEX_NEURALNET_PBACTORCRITIC_NET_H_

#include <vector>

#include "ActorCriticOutput.h"
#include "Action.h"
#include "PBActorNet.h"
#include "AdaptiveCriticNet3.h"

using namespace std;

namespace flexnnet
{

   class PBActorCriticNet
   {
   public:
      PBActorCriticNet (PBActorNet *_actor, AdaptiveCriticNet3 *_adaptCritic);
      virtual ~PBActorCriticNet ();

      void set_stochastic_action (bool _val);
      void set_print_gradient (bool _val);

      PBActorNet *get_actor ();
      AdaptiveCriticNet3 *get_adaptive_critic ();
      bool get_stochastic_action ();

      const ActorCriticOutput &operator() (const Pattern &ipattern, unsigned int recurStep = 1);
      const ActorCriticOutput &
      operator() (const Pattern &ipattern, const Pattern &apattern, unsigned int recurStep = 1);

      void clear_error (unsigned int timeStep = 1);
      void backprop (const vector<double> &_eVec, unsigned int timeStep = 1);

   private:
      PBActorNet *actor;
      AdaptiveCriticNet3 *adaptive_critic;

      ActorCriticOutput last_actor_critic_output;

      bool stochastic_action;
      bool print_gradient;
   };

   inline
   void PBActorCriticNet::set_stochastic_action (bool _val)
   {
      stochastic_action = _val;
   }

   inline
   bool PBActorCriticNet::get_stochastic_action ()
   {
      return stochastic_action;
   }

   inline
   void PBActorCriticNet::set_print_gradient (bool _val)
   {
      print_gradient = _val;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_ACTORCRITIC_NET3_H_ */
