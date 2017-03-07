/*
 * ActorCriticOutput.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#include <ActorCriticOutput.h>

namespace flex_neuralnet
{

ActorCriticOutput::ActorCriticOutput()
{
   reinforcement_signal = 0;
   reinforcement_estimated_flag = true;
}

ActorCriticOutput::ActorCriticOutput(const Action& _action, double _rSig, bool _estFlag)
{
   set(_action, _rSig, _estFlag);
}

ActorCriticOutput::~ActorCriticOutput()
{
}

void ActorCriticOutput::set(const Action& _action, double _rSig, bool _estFlag)
{
   actor_action = _action;
   reinforcement_signal = _rSig;
   reinforcement_estimated_flag = _estFlag;
}

ActorCriticOutput::ActorCriticOutput(const ActorCriticOutput& _acOut)
{
   copy(_acOut);
}

const Action& ActorCriticOutput::action() const
{
   return actor_action;
}

const double ActorCriticOutput::reinforcement() const
{
   return reinforcement_signal;
}

const bool ActorCriticOutput::is_reinforcement_estimated() const
{
   return reinforcement_estimated_flag;
}

ActorCriticOutput& ActorCriticOutput::operator=(const ActorCriticOutput& _acOut)
{
   copy(_acOut);
   return *this;
}


void ActorCriticOutput::copy(const ActorCriticOutput& _acOut)
{
   set(_acOut.action(), _acOut.reinforcement(), _acOut.is_reinforcement_estimated());
}

} /* namespace flex_neuralnet */
