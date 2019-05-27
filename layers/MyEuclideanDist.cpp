//
// Created by kfedrick on 5/24/19.
//

#include "MyEuclideanDist.h"

#include <cmath>

using namespace flexnnet;

MyEuclideanDist::MyEuclideanDist(unsigned int _netinvec_sz) : netinvec_size(_netinvec_sz)
{
   rawinvec_size = 0;
   stale_flag = true;

   netin.resize(netinvec_size);
}

MyEuclideanDist::~MyEuclideanDist()
{

}


/**
 * Calculate the net input value based on the raw input std::vector and weights specified in the
 * argument list and writes it into the _netin argument.
 */
const std::vector<double>&
MyEuclideanDist::calc_netin (const std::vector<double> &_rawin, const Array<double> &_weights)
{
   static double temp;
   squared_euclidean_dist.resize(netin.size());
   spread.resize(netin.size());


   if (_rawin.size () + 1 != _weights.colDim ())
      throw std::invalid_argument ("input std::vector size doesn't match column dimension in weight array");

   if (netin.size () != _weights.rowDim ())
      throw std::invalid_argument ("net input std::vector size doesn't match row dimension in weight array");

   /*
    * Calculate actvity to the netInNdx neuron as the euclidean distance
    * from the input vector to the vector specified by the weights vector
    * to the netInNdx neuron. Use the bias as a spread parameter.
    */
   unsigned int bias_ndx = _rawin.size ();
   for (unsigned int netInNdx = 0; netInNdx < netin.size (); netInNdx++)
   {
      squared_euclidean_dist[netInNdx] = 0;
      for (unsigned int rawInNdx = 0; rawInNdx < _rawin.size (); rawInNdx++)
      {
         temp = _rawin.at (rawInNdx) - _weights[netInNdx][rawInNdx];
         squared_euclidean_dist[netInNdx] += temp * temp;
      }

      spread[netInNdx] = exp (-_weights[netInNdx][bias_ndx]);
      netin.at (netInNdx) = spread[netInNdx] * squared_euclidean_dist[netInNdx];
   }

   return netin;
}

/**
 * Calculate the derivative of the net input with respect to the weights based on the raw
 * input std::vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
const Array<double>&
MyEuclideanDist::calc_dNdW (const std::vector<double> &_netin, const std::vector<double> &_rawin, const Array<
   double> &_weights)
{
   if (_rawin.size () + 1 != _weights.colDim ())
      throw std::invalid_argument ("external input vector size doesn't match column dimension in weight array");

   if (_netin.size () != _weights.rowDim ())
      throw std::invalid_argument ("net input vector size doesn't match row dimension in weight array");

   if (dNdW.rowDim () != _netin.size () || dNdW.colDim () != _rawin.size () + 1)
      throw std::invalid_argument ("dNdW array dimensionality doesn't match weight array");

   unsigned int spread_param_ndx = _rawin.size();
   for (unsigned int out_ndx = 0; out_ndx < _netin.size (); out_ndx++)
   {
      // Calculate dNdW with respect to the spread parameter
      dNdW[out_ndx][spread_param_ndx] += -squared_euclidean_dist[out_ndx] * spread[out_ndx];

      // Calculate dNdW with respect to RBF kernel
      for (unsigned int in_ndx = 0; in_ndx < _rawin.size (); in_ndx++)
         dNdW[out_ndx][in_ndx] = -2.0 * spread[out_ndx] * (_rawin.at (in_ndx) - _weights[out_ndx][in_ndx]);

      return dNdW;
   }
}

/**
 * Calculate the derivative of the net input with respect to the raw input based on the raw
 * input vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
const Array<double>&
MyEuclideanDist::calc_dNdI (const std::vector<double> &_netin, const std::vector<double> &_rawin, const Array<
   double> &_weights)
{

   if (_rawin.size () + 1 != _weights.colDim ())
      throw std::invalid_argument ("input std::vector size doesn't match column dimension in weight array");

   if (dNdI.rowDim () != _netin.size () || dNdI.colDim () != _rawin.size () + 1)
      throw std::invalid_argument ("dNdI array dimensionality doesn't match weight array");

   unsigned int spread_param_ndx = _rawin.size();
   for (unsigned int out_ndx = 0; out_ndx < _netin.size (); out_ndx++)
   {
      // TODO - probably not needed
      dNdI[out_ndx][spread_param_ndx] = 0;

      // Calculate dNdI with respect to input RBF kernel
      for (unsigned int in_ndx = 0; in_ndx < _rawin.size (); in_ndx++)
         dNdI[out_ndx][in_ndx] = 2.0 * spread[out_ndx] * (_rawin.at (in_ndx) - _weights[out_ndx][in_ndx]);

      return dNdI;
   }
}


