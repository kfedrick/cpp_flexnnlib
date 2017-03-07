/*
 * NetSum.cpp
 *
 *  Created on: Feb 1, 2014
 *      Author: kfedrick
 */

#include "NetSum.h"

#include <stdio.h>
#include <stdexcept>

using namespace std;

namespace flex_neuralnet
{

NetSum::NetSum() : NetInputFunctor("NetSum")
{
}

/*
 * Calculate the net input and gradients given the raw input and weight array using a weighted the weighted
 * sum of the raw input vector.
 *
 * Arguments:
 *
 *    netInVec (out) : the calculated net input vector
 *    dNdW (out)     : the derivative of the net input vector with respect to the weight array
 *    dNdI (out)     : the derivative of the net input vector with respect to the raw input vector
 *    rawInVec (in)  : the raw input vector
 *    weights (in)   : the layer weight array
 *
 * Preconditions:
 *
 * 1. Weights is an M x N array where M is the dimensionality of the raw input vector,
 *    rawInVec, and N is the dimensionality of the net input vector, netInVec
 *
 * 2. The array dNdW is the same dimensionality as the weight array..
 *
 * 3. The array dNdI has the same dimensionality as the weight array.
 */
void NetSum::operator()(vector<double>& netInVec, Array<double>& dNdW, Array<double>& dNdI,
      const vector<double>& rawInVec, const Array<double>& weights) const
{

   if (rawInVec.size() != weights.colDim())
      throw invalid_argument("input vector size doesn't match column dimension in weight array");

   if (netInVec.size() != weights.rowDim())
      throw invalid_argument("net input vector size doesn't match row dimension in weight array");

   if (dNdW.rowDim() != weights.rowDim() || dNdW.colDim() != weights.colDim())
      throw invalid_argument("dNdW array dimensionality doesn't match weight array");

   if (dNdI.rowDim() != weights.rowDim() || dNdI.colDim() != weights.colDim())
      throw invalid_argument("dNdI array dimensionality doesn't match weight array");


   for (unsigned int netInNdx = 0; netInNdx < netInVec.size(); netInNdx++)
   {
      netInVec.at(netInNdx) = 0;
      for (unsigned int rawInNdx = 0; rawInNdx < rawInVec.size(); rawInNdx++)
          netInVec.at(netInNdx) += rawInVec.at(rawInNdx) * weights[netInNdx][rawInNdx];
   }

   for (unsigned int out_ndx=0; out_ndx<dNdW.rowDim(); out_ndx++)
       for (unsigned int in_ndx=0; in_ndx<rawInVec.size(); in_ndx++)
           dNdW[out_ndx][in_ndx] = rawInVec.at(in_ndx);

   for (unsigned int out_ndx=0; out_ndx<dNdI.rowDim(); out_ndx++)
      for (unsigned int in_ndx=0; in_ndx<rawInVec.size(); in_ndx++)
         dNdI[out_ndx][in_ndx] = weights[out_ndx][in_ndx];

}

NetSum* NetSum::clone() const
{
   return new NetSum(*this);
}

} /* namespace flex_neuralnet */
