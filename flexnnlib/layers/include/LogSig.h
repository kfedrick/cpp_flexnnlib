/*
 * LogSig.h
 *
 *  Created on: Mar 10, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_LOGSIG_H_
#define FLEX_NEURALNET_LOGSIG_H_

#include "TransferFunctor.h"
#include <cmath>

namespace flex_neuralnet
{

class LogSig: public flex_neuralnet::TransferFunctor
{
public:
   LogSig();
   virtual ~LogSig();

   double get_gain() const;
   void set_gain(double val);

   void operator()(vector<double>& transVec, Array<double>& dAdN, vector<double>& d2AdN,
         Array<double>& dAdB, const vector<double>& netInVec, const vector<double>& biases) const;

   virtual LogSig* clone() const;

private:
   double gain;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_LOGSIG_H_ */
