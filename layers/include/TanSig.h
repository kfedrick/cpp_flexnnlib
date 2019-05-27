/*
 * TanSig.h
 *
 *  Created on: Mar 10, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_TANSIG_H_
#define FLEX_NEURALNET_TANSIG_H_

#include "TransferFunctor.h"
#include "NetSum.h"
#include <cmath>

namespace flexnnet
{

   class TanSig : public TransferFunctor, public NetSum
   {
   public:

      /*
       * Return transfer function type for default object name creation.
       */
      static string type()
      {
         return "TanSig";
      }

      TanSig ();

      double get_gain () const;
      void set_gain (double val);

      void operator() (vector<double> &transVec, Array<double> &dAdN, vector<double> &d2AdN,
                       Array<double> &dAdB, const vector<double> &netInVec, const vector<double> &biases) const;

      TanSig *clone () const;

   private:
      double gain;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_TANSIG_H_ */
