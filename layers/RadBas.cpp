/*
 * RadBas.cpp
 *
 *  Created on: Mar 31, 2014
 *      Author: kfedrick
 */

#include "RadBas.h"
#include <cmath>

namespace flex_neuralnet
{

RadBas::RadBas() : TransferFunctor("RadBas")
{
}

void RadBas::operator()(vector<double>& transVec, Array<double>& dAdN, vector<double>& d2AdN,
      Array<double>& dAdB, const vector<double>& netInVec, const vector<double>& biasVec) const
{
   dAdN = 0;
   dAdB = 0;
   for (unsigned int i=0; i<transVec.size(); i++)
   {
      sqr_dist = netInVec[i] * netInVec[i];
      spread = exp( -biasVec[i] );

      transVec[i] = exp( -spread * sqr_dist );
      dAdN[i][i] = -2.0 * spread * netInVec[i] * transVec[i];
      dAdB[i][i] = spread * sqr_dist * transVec[i];
   }

}

RadBas* RadBas::clone() const
{
   return new RadBas(*this);
}

}
