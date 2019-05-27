/*
 * OutputErrorFunctor.cpp
 *
 *  Created on: Feb 5, 2014
 *      Author: kfedrick
 */

#include "OutputErrorFunctor.h"
#include <iostream>

using namespace std;

namespace flexnnet
{

   OutputErrorFunctor::OutputErrorFunctor ()
   {
      // TODO Auto-generated constructor stub
   }

   OutputErrorFunctor::~OutputErrorFunctor ()
   {
      // TODO Auto-generated destructor stub
   }

   void
   OutputErrorFunctor::operator() (double &error, vector<double> &gradient, const vector<double> &outVec, const vector<
      double> &targetVec)
   {

   }

   OutputErrorFunctor *OutputErrorFunctor::clone () const
   {
      return new OutputErrorFunctor (*this);
   }

}
