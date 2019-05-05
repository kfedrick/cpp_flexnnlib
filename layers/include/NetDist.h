/*
 * NetDist.h
 *
 *  Created on: Mar 31, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_NETDIST_H_
#define FLEX_NEURALNET_NETDIST_H_

#include "NetInputFunctor.h"
#include <cmath>

namespace flex_neuralnet
{

class NetDist: public NetInputFunctor
{
public:
   NetDist();

   /*
    * Calculate the net input value based on the raw input vector and weights specified in the
    * argument list and copies it into the netInVec argument.
    */
   void operator()(vector<double>& netInVec, Array<double>& dNdW, Array<double>& dNdI,
         const vector<double>& rawInVec, const Array<double>& weights) const;

   NetDist* clone() const;

private:
   mutable double temp, temp_sum, temp_deriv;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_NETDIST_H_ */
