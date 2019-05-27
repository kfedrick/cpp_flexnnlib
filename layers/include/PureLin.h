/*
 * TransLin.h
 *
 *  Created on: Feb 1, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_TRANSLIN_H_
#define FLEX_NEURALNET_TRANSLIN_H_

#include "TransferFunctor.h"
#include "NetSum.h"

namespace flexnnet
{

   class PureLin : public TransferFunctor, public NetSum
   {

   public:
      static string type()
      {
         return "PureLin";
      }

   public:
      PureLin ();

      void operator() (vector<double> &transVec, Array<double> &dAdN, vector<double> &d2AdN,
                       Array<double> &dAdB, const vector<double> &netInVec, const vector<double> &biases) const;

      PureLin *clone () const;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_TRANSLIN_H_ */
