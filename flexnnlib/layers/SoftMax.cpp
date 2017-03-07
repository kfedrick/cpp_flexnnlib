/*
 * SoftMax.cpp
 *
 *  Created on: Mar 10, 2015
 *      Author: kfedrick
 */

#include "SoftMax.h"
#include <cmath>

using namespace std;
namespace flex_neuralnet
{

SoftMax::SoftMax()
{
   set_gain(1.0);
   set_output_range(0.0, 1.0);
}

SoftMax::~SoftMax()
{
   // No additional work required
}

double SoftMax::get_gain() const
{
   return gain;
}

void SoftMax::get_output_range(double& lower, double& upper) const
{
   lower = lower_bound;
   upper = upper_bound;
}

void SoftMax::set_gain(double val)
{
   gain = val;
}

void SoftMax::set_output_range(double lower, double upper)
{
   if (lower > upper)
      throw invalid_argument(
            "SoftMax::set_output_range(lower, upper) : lower bound must be less than upper bound.");

   lower_bound = lower;
   upper_bound = upper;
   output_range = upper_bound - lower_bound;
}

void SoftMax::operator()(vector<double>& transVec, Array<double>& dAdN,
      Array<double>& dAdB, const vector<double>& netInVec,
      const vector<double>& biasVec) const
{
   double sum_exp = 0;

   // Calculate the initial exp of the input values and accumulate the summation
   for (unsigned int i = 0; i < transVec.size(); i++)
   {
      transVec[i] = exp(gain * (biasVec[i] + netInVec[i]));
      sum_exp += transVec[i];
   }

   // Normalize the transfer vector by dividing by the sum of activity
   for (unsigned int i = 0; i < transVec.size(); i++)
      transVec[i] = output_range * transVec[i] / sum_exp + lower_bound;

   dAdN = 0;
   for (unsigned int netin_ndx = 0; netin_ndx < netInVec.size(); netin_ndx++)
   {
      for (unsigned int trans_ndx = 0; trans_ndx < transVec.size(); trans_ndx++)
         dAdN[trans_ndx][netin_ndx] =
               (trans_ndx == netin_ndx) ?
                     output_range * gain * transVec[trans_ndx] * (1 - transVec[trans_ndx]) :
                     output_range * gain * -transVec[trans_ndx] * transVec[netin_ndx];
   }

   dAdB = 0;
   for (unsigned int netin_ndx = 0; netin_ndx < netInVec.size(); netin_ndx++)
   {
      for (unsigned int trans_ndx = 0; trans_ndx < transVec.size(); trans_ndx++)
         dAdB[trans_ndx][netin_ndx] =
               (trans_ndx == netin_ndx) ?
                     output_range * gain * transVec[trans_ndx] * (1 - transVec[trans_ndx]) :
                     output_range * gain * -transVec[trans_ndx] * transVec[netin_ndx];
   }

   /*
    * Partial of the i'th value in the transfer vector wrt the gain
    * = transVec[i] * (netInVec[i] - sum(netInVec[j] * transVec[j]) over all j
    */
}

SoftMax* SoftMax::clone() const
{
   return new SoftMax(*this);
}

} /* namespace flex_neuralnet */
