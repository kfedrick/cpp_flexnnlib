/*
 * RadBas.cpp
 *
 *  Created on: Mar 31, 2014
 *      Author: kfedrick
 */

#include "RadBas.h"
#include <cmath>

namespace flexnnet
{

   RadBas::RadBas () : TransferFunctor ("RadBas")
   {
   }

   void RadBas::operator() (vector<double> &transVec, Array<double> &dAdN, vector<double> &d2AdN,
                            Array<double> &dAdB, const vector<double> &netInVec, const vector<double> &biasVec) const
   {
      dAdN = 0;
      dAdB = 0;
      for (unsigned int i = 0; i < transVec.size (); i++)
      {
         sqr_dist = netInVec[i] * netInVec[i];
         spread = exp (-biasVec[i]);

         transVec[i] = exp (-spread * sqr_dist);
         dAdN[i][i] = -2.0 * spread * netInVec[i] * transVec[i];
         // If sqrt(spread) part of netin netin then
         //    dAdN = -2.0 * netInVec[i] * transVec[i]
         dAdB[i][i] = spread * sqr_dist * transVec[i];
         // If sqrt(spread) part of netin then
         //    dNdB = old netin * 1/2 (1/sqrt(exp(-bias))) * exp(-bias) * (-1)
         //         = old netin * -1/2 sqrt(exp(-bias))
         //    dAdB = dAdN * dNdB = 1 * netInVec * transVec * sqrt(dist) * sqrt(exp(-bias))
         //         = netInVec * transVec * netInVec
         //         = netInVec^2 * transVec
      }

   }

   RadBas *RadBas::clone () const
   {
      return new RadBas (*this);
   }

}
