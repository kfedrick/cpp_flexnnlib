/*
 * ArrayInitializer.h
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ARRAYINITI_H_
#define FLEX_NEURALNET_ARRAYINITI_H_

#include <Array.h>

namespace flex_neuralnet
{

class ArrayInitializer
{
public:

   virtual ~ArrayInitializer();

   virtual void operator()(Array<double>& arr) const;

   virtual ArrayInitializer* clone() const;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_ARRAYINITI_H_ */
