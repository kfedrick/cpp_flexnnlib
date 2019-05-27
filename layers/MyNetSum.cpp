//
// Created by kfedrick on 5/11/19.
//

#include "MyNetSum.h"

using namespace flexnnet;

MyNetSum::MyNetSum(unsigned int _netinvec_sz) : netinvec_size(_netinvec_sz)
{
   rawinvec_size = 0;

   netin.resize(netinvec_size);
}

MyNetSum::~MyNetSum()
{

}


/**
 * Calculate the net input value based on the raw input std::vector and weights specified in the
 * argument list and writes it into the _netin argument.
 */
const std::vector<double>& MyNetSum::calc_netin (const std::vector<double> &_rawin, const Array<double>& _weights)
{
   if (_rawin.size ()+1 != _weights.colDim ())
      throw std::invalid_argument ("input std::vector size doesn't match column dimension in weight array");

   if (netin.size () != _weights.rowDim ())
      throw std::invalid_argument ("net input std::vector size doesn't match row dimension in weight array");

   unsigned int bias_ndx = _rawin.size();
   for (unsigned int netInNdx = 0; netInNdx < netin.size (); netInNdx++)
   {
      // Initialize netin value with bias, then add weighted sum of input vector
      netin.at (netInNdx) = _weights[netInNdx][bias_ndx];
      for (unsigned int rawInNdx = 0; rawInNdx < _rawin.size (); rawInNdx++)
         netin.at (netInNdx) += _rawin.at (rawInNdx) * _weights[netInNdx][rawInNdx];
   }

   return netin;
}

/**
 * Calculate the derivative of the net input with respect to the weights based on the raw
 * input std::vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
const Array<double>& MyNetSum::calc_dNdW (const std::vector<double> &_netin, const std::vector<double> &_rawin, const Array<double>& _weights)
{
   if (_rawin.size () + 1 != _weights.colDim ())
      throw std::invalid_argument ("external input vector size doesn't match column dimension in weight array");

   if (_netin.size () != _weights.rowDim ())
      throw std::invalid_argument ("net input vector size doesn't match row dimension in weight array");

   if (dNdW.rowDim () != _netin.size () || dNdW.colDim () != _rawin.size()+1)
      throw std::invalid_argument ("dNdW array dimensionality doesn't match weight array");

   unsigned int bias_ndx = _rawin.size();
   for (unsigned int out_ndx = 0; out_ndx < _netin.size(); out_ndx++)
   {
      /*
      _dNdW[out_ndx][bias_ndx] = 1;
      for (unsigned int in_ndx = 0; in_ndx < _rawin.size (); in_ndx++)
         _dNdW[out_ndx][in_ndx] = _rawin.at (in_ndx);
      */

      dNdW[out_ndx][bias_ndx] = 1;
      for (unsigned int in_ndx = 0; in_ndx < _rawin.size (); in_ndx++)
         dNdW[out_ndx][in_ndx] = _rawin.at (in_ndx);

      return dNdW;
   }
}

/**
 * Calculate the derivative of the net input with respect to the raw input based on the raw
 * input vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
const Array<double>& MyNetSum::calc_dNdI (const std::vector<double> &_netin, const std::vector<double> &_rawin, const Array<double>& _weights)
{
   if (_rawin.size ()+1 != _weights.colDim ())
      throw std::invalid_argument ("input std::vector size doesn't match column dimension in weight array");

   if (dNdI.rowDim () != _netin.size() || dNdI.colDim () != _rawin.size()+1)
      throw std::invalid_argument ("dNdI array dimensionality doesn't match weight array");

   unsigned int bias_ndx = _rawin.size();
   for (unsigned int out_ndx = 0; out_ndx < _netin.size(); out_ndx++)
   {
      dNdI[out_ndx][bias_ndx] = _weights[out_ndx][bias_ndx];
      for (unsigned int in_ndx = 0; in_ndx < _rawin.size (); in_ndx++)
         dNdI[out_ndx][in_ndx] = _weights[out_ndx][in_ndx];

      return dNdI;
   }
}
