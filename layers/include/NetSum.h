/*
 * NetSum.h
 *
 *  Created on: Feb 1, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_NETSUM_H_
#define FLEX_NEURALNET_NETSUM_H_

#include "NetInputFunctor.h"

namespace flexnnet
{

   class NetSum : public NetInputFunctor
   {
   public:

      NetSum ();

      /*
       * Calculate the net input value based on the raw input vector and weights specified in the
       * argument list and copies it into the netInVec argument.
       */
      void operator() (vector<double> &netInVec, Array<double> &dNdW, Array<double> &dNdI,
                       const vector<double> &rawInVec, const Array<double> &weights) const;

      NetSum *clone () const;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_NETSUM_H_ */
