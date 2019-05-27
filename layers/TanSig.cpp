/*
 * TanSig.cpp
 *
 *  Created on: Mar 10, 2014
 *      Author: kfedrick
 */

#include "TanSig.h"
#include <iostream>

using namespace std;

namespace flexnnet
{

   TanSig::TanSig () : TransferFunctor ("TanSig")
   {
      gain = 1.0;
   }

   double TanSig::get_gain () const
   {
      return gain;
   }

   void TanSig::set_gain (double val)
   {
      gain = val;
   }

   void TanSig::operator() (vector<double> &transVec, Array<double> &dAdN, vector<double> &d2AdN,
                            Array<double> &dAdB, const vector<double> &netInVec, const vector<double> &biasVec) const
   {
      for (unsigned int i = 0; i < transVec.size (); i++)
         transVec[i] = 2.0 / (1.0 + exp (-2.0 * gain * (biasVec[i] + netInVec[i]))) - 1.0;

      dAdN = 0;
      for (unsigned int i = 0; i < transVec.size (); i++)
      {
         dAdN[i][i] = gain * (1 - transVec[i] * transVec[i]);
         d2AdN[i] = -transVec[i] * (1 - transVec[i] * transVec[i]);
      }

      dAdB = 0;
      for (unsigned int i = 0; i < transVec.size (); i++)
         dAdB[i][i] = gain * (1 - transVec[i] * transVec[i]);
   }

   TanSig *TanSig::clone () const
   {
      return new TanSig (*this);
   }

} /* namespace flexnnet */
