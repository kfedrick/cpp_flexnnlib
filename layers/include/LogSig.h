/*
 * LogSig.h
 *
 *  Created on: Mar 10, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_LOGSIG_H_
#define FLEX_NEURALNET_LOGSIG_H_

#include "TransferFunctor.h"
#include "NetSum.h"
#include <cmath>

namespace flexnnet
{

   class LogSig : public TransferFunctor, public NetSum
   {
   public:

      /*
       * Return transfer function type for default object name creation.
       */
      static string type()
      {
         return "LogSig";
      }

      LogSig ();
      virtual ~LogSig ();

      double get_gain () const;
      void set_gain (double val);

      void operator() (vector<double> &transVec, Array<double> &dAdN, vector<double> &d2AdN,
                       Array<double> &dAdB, const vector<double> &netInVec, const vector<double> &biases) const;

      virtual LogSig *clone () const;

   private:
      double gain;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_LOGSIG_H_ */
