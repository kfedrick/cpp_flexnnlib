/*
 * ActorCriticOutput.h
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ACTORCRITIC_OUTPUT_H_
#define FLEX_NEURALNET_ACTORCRITIC_OUTPUT_H_

#include "Action.h"

using namespace std;

namespace flex_neuralnet
{

class ActorCriticOutput
{
public:
   ActorCriticOutput();
   ActorCriticOutput(const Action& _action, double _rSig, bool _estFlag=true);
   ActorCriticOutput(const ActorCriticOutput& _acOut);
   virtual ~ActorCriticOutput();

   const Action& action() const;
   const double reinforcement() const;
   const bool is_reinforcement_estimated() const;

   ActorCriticOutput& operator=(const ActorCriticOutput& _acOut);

   void set(const Action& _action, double _rSig, bool _estFlag);

private:
   void copy(const ActorCriticOutput& _acOut);

private:
   Action actor_action;
   double reinforcement_signal;
   bool reinforcement_estimated_flag;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_ACTORCRITIC_OUTPUT_H_ */
