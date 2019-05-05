/*
 * TransLin.h
 *
 *  Created on: Feb 1, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_TRANSLIN_H_
#define FLEX_NEURALNET_TRANSLIN_H_

#include "TransferFunctor.h"

namespace flex_neuralnet
{

class PureLin : public TransferFunctor
{
public:
   PureLin();

   void operator()(vector<double>& transVec, Array<double>& dAdN, vector<double>& d2AdN,
         Array<double>& dAdB, const vector<double>& netInVec, const vector<double>& biases) const;

   PureLin* clone() const;
};



} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_TRANSLIN_H_ */
