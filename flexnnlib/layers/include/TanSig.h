/*
 * TanSig.h
 *
 *  Created on: Mar 10, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_TANSIG_H_
#define FLEX_NEURALNET_TANSIG_H_

#include "TransferFunctor.h"
#include <cmath>

namespace flex_neuralnet
{

class TanSig: public TransferFunctor
{
public:
   TanSig();

   double get_gain() const;
   void set_gain(double val);

   void operator()(vector<double>& transVec, Array<double>& dAdN, vector<double>& d2AdN,
         Array<double>& dAdB, const vector<double>& netInVec, const vector<double>& biases) const;

   TanSig* clone() const;

private:
   double gain;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_TANSIG_H_ */
