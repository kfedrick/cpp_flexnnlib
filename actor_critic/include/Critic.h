/*
 * Critic.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_CRITIC_H_
#define FLEX_NEURALNET_CRITIC_H_

#include <vector>
#include "Pattern.h"
#include "Action.h"

using namespace std;

namespace flexnnet
{

   class Critic
   {
   public:
      virtual ~Critic ();

      virtual double reinforcement (const Pattern &_state,
                                    const Action &_action, bool &rFlag) = 0;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_CRITIC_H_ */
