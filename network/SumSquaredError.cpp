/*
 * MeanSquaredError.cpp
 *
 *  Created on: Feb 19, 2014
 *      Author: kfedrick
 */

#include "SumSquaredError.h"
#include <stdexcept>
#include <iostream>

using namespace std;

namespace flex_neuralnet
{

SumSquaredError::SumSquaredError()
{
   // TODO Auto-generated constructor stub
}

SumSquaredError::~SumSquaredError()
{
   // TODO Auto-generated destructor stub
}

void SumSquaredError::operator()(double& error, vector<double>& gradient, const vector<double>& outVec, const vector<double>& targetVec)
{

   unsigned int sz = targetVec.size();

   if (outVec.size() != sz)
      throw invalid_argument("SumSquaredError::operator() Error - output vector and target vector must be the same size.");

   if (gradient.size() != sz)
      throw invalid_argument("SumSquaredError::operator() Error - output vector and gradient vector must be the same size.");

   double diff;
   error = 0;
   for (unsigned int ndx=0; ndx < sz; ndx++)
   {
       diff = -(targetVec[ndx] - outVec[ndx]);

       gradient[ndx] = diff;
       error += diff * diff;
   }

   error *= 0.5;
}

SumSquaredError* SumSquaredError::clone() const
{
   return new SumSquaredError(*this);
}

}
