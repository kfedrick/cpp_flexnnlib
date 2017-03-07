/*
 * AdaptiveComponent.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ADAPTIVE_COMPONENT_H_
#define FLEX_NEURALNET_ADAPTIVE_COMPONENT_H_

#include <vector>

using namespace std;

namespace flex_neuralnet
{

class AdaptiveComponent
{
public:
   virtual void clear_reinforcement() = 0;
   virtual void accumulate_reinforcement(const double _estRSig);
   virtual void accumulate_reinforcement(const double _estRSig, const double _extRSig);
   virtual void adapt() = 0;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_ADAPTIVE_COMPONENT_H_ */
