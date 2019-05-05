/*
 * NetInputFunctor.h
 *
 *  Created on: Feb 1, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_NETINPUT_H_
#define FLEX_NEURALNET_NETINPUT_H_

#include <vector>
#include "NamedObject.h"
#include "Array.h"

using namespace std;

namespace flex_neuralnet
{

class NetInputFunctor : public NamedObject
{
public:

   NetInputFunctor(const string& name = "NetInputFunctor") : NamedObject(name) {}
   NetInputFunctor(const char* name) : NamedObject(name) {}

   virtual void operator()(vector<double>& netInVec, Array<double>& dAdW, Array<double>& dNdI,
         const vector<double>& rawInVec, const Array<double>& weights) const = 0;

   virtual NetInputFunctor* clone() const = 0;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_NETINPUT_H_ */
