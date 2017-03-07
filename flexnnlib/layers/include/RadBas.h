/*
 * RadBas.h
 *
 *  Created on: Mar 31, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_RADBAS_H_
#define FLEX_NEURALNET_RADBAS_H_

#include "TransferFunctor.h"

namespace flex_neuralnet
{

class RadBas: public TransferFunctor
{
public:
   RadBas();

   void operator()(vector<double>& transVec, Array<double>& dAdN,
         Array<double>& dAdB, const vector<double>& netInVec, const vector<double>& biases) const;

   RadBas* clone() const;

private:
   mutable double spread;
   mutable double sqr_dist;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_RADBAS_H_ */
