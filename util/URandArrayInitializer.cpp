/*
 * URandArrayInitializer.cpp
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#include "URandArrayInitializer.h"
#include <stdio.h>

namespace flexnnet
{

   URandArrayInitializer::URandArrayInitializer (double lower, double upper)
   {
      lower_bound = lower;
      upper_bound = upper;

      srand (time (NULL));
   }

   URandArrayInitializer::~URandArrayInitializer ()
   {
      // TODO Auto-generated destructor stub
   }

   double URandArrayInitializer::urand () const
   {
      return rand () / double (RAND_MAX);
   }

   double URandArrayInitializer::urand (double a, double b) const
   {
      return (b - a) * urand () + a;
   }

   void URandArrayInitializer::operator() (Array<double> &arr) const
   {
      arr.set (0);
      for (int i = 0; i < arr.rowDim (); i++)
         for (int j = 0; j < arr.colDim (); j++)
            arr.at (i, j) = urand (lower_bound, upper_bound);
   }

   URandArrayInitializer *URandArrayInitializer::clone () const
   {
      return new URandArrayInitializer (*this);
   }

} /* namespace flexnnet */
