/*
 * TransLin.cpp
 *
 *  Created on: Feb 1, 2014
 *      Author: kfedrick
 */

#include "PureLin.h"

namespace flexnnet
{

   PureLin::PureLin () : TransferFunctor ("PureLin")
   {
   }

   void PureLin::operator() (vector<double> &transVec, Array<double> &dAdN, vector<double> &d2AdN,
                             Array<double> &dAdB, const vector<double> &netInVec, const vector<double> &biasVec) const
   {
      dAdN = 0;
      dAdB = 0;
      for (unsigned int i = 0; i < transVec.size (); i++)
      {
         transVec[i] = biasVec[i] + netInVec[i];
         dAdN[i][i] = 1;
         dAdB[i][i] = 1;

         d2AdN[i] = 0;
      }
   }

   PureLin *PureLin::clone () const
   {
      return new PureLin (*this);
   }



} /* namespace flexnnet */
