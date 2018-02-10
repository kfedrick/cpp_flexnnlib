/*
 * TransferFunctor.h
 *
 *  Created on: Feb 1, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_TRANSFERFUNCTOR_H_
#define FLEX_NEURALNET_TRANSFERFUNCTOR_H_

#include "NamedObject.h"
#include <vector>
#include "Array.h"

using namespace std;

namespace flex_neuralnet
{

class TransferFunctor : public NamedObject
{
public:
   TransferFunctor(const string& name="TransferFunctor") : NamedObject(name) {}
   TransferFunctor(const char* name) : NamedObject(name) {}
   virtual ~TransferFunctor() {}

   virtual void operator()(vector<double>& transVec, Array<double>& dAdN, vector<double>& d2AdN,
         Array<double>& dAdB, const vector<double>& netInVec, const vector<double>& biases) const = 0;

   virtual TransferFunctor* clone() const = 0;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_TRANSFERFUNCTOR_H_ */
