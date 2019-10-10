/*
 * RMSError.h
 *
 *  Created on: Feb 19, 2014
 *      Author: kfedrick
 */

#ifndef SUMSQUAREDERROR_H_
#define SUMSQUAREDERROR_H_

#include "OutputErrorFunctor.h"

namespace flexnnet
{

   class SumSquaredError : public flexnnet::OutputErrorFunctor
   {
   public:
      SumSquaredError ();
      virtual ~SumSquaredError ();

      void operator() (double &error, vector<double> &gradient,
                       const vector<double> &outVec, const vector<double> &targetVec);

      SumSquaredError *clone () const;
   };

}

#endif /* SUMSQUAREDERROR_H_ */
