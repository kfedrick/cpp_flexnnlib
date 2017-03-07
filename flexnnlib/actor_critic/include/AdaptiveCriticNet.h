/*
 * AdaptiveCriticNet.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ADAPTIVECRITIC_NET_H_
#define FLEX_NEURALNET_ADAPTIVECRITIC_NET_H_

#include "BaseNeuralNet.h"

using namespace std;

namespace flex_neuralnet
{

class AdaptiveCriticNet: public BaseNeuralNet
{
public:
   AdaptiveCriticNet();
   virtual ~AdaptiveCriticNet();

   virtual double get_reinforcement(const Pattern& _stateVec,
         const Pattern& _actionVec, unsigned int recurStep = 1);

private:
   ConnectionMap conn_map;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_ADAPTIVECRITIC_NET_H_ */
