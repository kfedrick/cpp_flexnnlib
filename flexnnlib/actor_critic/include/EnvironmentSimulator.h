/*
 * EnvironmentSimulator.h
 *
 *  Created on: Mar 18, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ENVIRONMENTSIMULATOR_H_
#define FLEX_NEURALNET_ENVIRONMENTSIMULATOR_H_

#include <vector>

#include "Action.h"
#include "Pattern.h"

using namespace std;

namespace flex_neuralnet
{

class EnvironmentSimulator
{
public:
   virtual const Pattern& reset() = 0;
   virtual const Pattern& next_state(const Action& _action) = 0;
   virtual bool is_terminal_state(const Pattern& _state) const = 0;
   virtual double get_reinforcement(bool& _rflag) = 0;

   // Provide hint for best action
   virtual const Pattern& hint();

protected:
   Pattern default_action;
};

inline
const Pattern& EnvironmentSimulator::hint(void)
{
   return default_action;
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_ENVIRONMENTSIMULATOR_H_ */
