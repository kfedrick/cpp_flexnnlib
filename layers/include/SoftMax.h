/*
 * SoftMax.h
 *
 *  Created on: Mar 10, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_SOFTMAX_H_
#define FLEX_NEURALNET_SOFTMAX_H_

#include "TransferFunctor.h"
#include "NetSum.h"
#include <cmath>

namespace flexnnet
{

   class SoftMax : public TransferFunctor, public NetSum
   {


   public:

      /*
       * Return transfer function type for default object name creation.
       */
      static string type()
      {
         return "SoftMax";
      }

      SoftMax ();
      virtual ~SoftMax ();

      double get_gain () const;
      void get_output_range (double &lower, double &upper) const;

      void set_gain (double val);
      void set_output_range (double lower, double upper);

      void operator() (vector<double> &transVec, Array<double> &dAdN, vector<double> &d2AdN,
                       Array<double> &dAdB, const vector<double> &netInVec, const vector<double> &biases) const;

      virtual SoftMax *clone () const;

   private:
      double gain;
      double lower_bound;
      double upper_bound;
      double output_range;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_SOFTMAX_H_ */
