/*
 * ActorCriticNet2.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ACTORCRITIC_NET2_H_
#define FLEX_NEURALNET_ACTORCRITIC_NET2_H_

#include <vector>

#include "ActorCriticOutput.h"
#include "Action.h"
#include "ActorNet2.h"
#include "AdaptiveCriticNet.h"

using namespace std;

namespace flex_neuralnet
{

class ActorCriticNet2
{
public:
   ActorCriticNet2(ActorNet2* _actor, AdaptiveCriticNet* _adaptCritic);
   virtual ~ActorCriticNet2();

   void set_stochastic_action(bool _val);
   void set_print_gradient(bool _val);

   ActorNet2* get_actor();
   AdaptiveCriticNet* get_adaptive_critic();
   bool get_stochastic_action();

   const ActorCriticOutput& operator()(const Pattern& ipattern, unsigned int recurStep = 1);
   const ActorCriticOutput& operator()(const Pattern& ipattern, const Pattern& apattern, unsigned int recurStep = 1);

   void clear_error(unsigned int timeStep = 1);
   void backprop(const vector<double>& _eVec, unsigned int timeStep = 1);

private:
   ActorNet2* actor;
   AdaptiveCriticNet* adaptive_critic;

   ActorCriticOutput last_actor_critic_output;

   bool stochastic_action;
   bool print_gradient;
};

inline
void ActorCriticNet2::set_stochastic_action(bool _val)
{
   stochastic_action = _val;
}

inline
bool ActorCriticNet2::get_stochastic_action()
{
   return stochastic_action;
}

inline
void ActorCriticNet2::set_print_gradient(bool _val)
{
   print_gradient = _val;
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_ACTORCRITIC_NET_H_ */
