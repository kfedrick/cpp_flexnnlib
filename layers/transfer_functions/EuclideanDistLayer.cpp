//
// Created by kfedrick on 6/10/19.
//

#include "EuclideanDistLayer.h"

using flexnnet::Array2D;
using flexnnet::EuclideanDistLayer;

EuclideanDistLayer::EuclideanDistLayer(size_t _sz, const std::string &_name, NetworkLayerType _type) : NetworkLayer(_sz, _name, _type)
{
   squared_euclidean_dist.resize(_sz);
   spread.resize(_sz);
}

EuclideanDistLayer::~EuclideanDistLayer()
{
}


/**
 * Calculate the net input value based on the raw input std::vector and weights specified in the
 * argument list and writes it into the _netin argument.
 */
const std::valarray<double>&
EuclideanDistLayer::calc_netin (const std::valarray<double> &_rawin)
{
   static double temp;

   const Array2D<double>& weights = layer_weights.const_weights_ref;
   Array2D<double>::Dimensions wdim = weights.size();

   std::valarray<double>& netinv = layer_state.netinv;


   if (_rawin.size () + 1 != wdim.cols)
      throw std::invalid_argument ("input vector size doesn't match column dimension in weight array");

   if (netinv.size () != wdim.rows)
      throw std::invalid_argument ("net input vector size doesn't match row dimension in weight array");

   /*
    * Calculate actvity to the netInNdx neuron as the euclidean distance
    * from the input vector to the vector specified by the weights vector
    * to the netInNdx neuron. Use the bias as a spread parameter.
    */
   size_t bias_ndx = _rawin.size ();
   for (size_t netInNdx = 0; netInNdx < const_layer_output_size_ref; netInNdx++)
   {
      squared_euclidean_dist[netInNdx] = 0;
      for (unsigned int rawInNdx = 0; rawInNdx < layer_input_size; rawInNdx++)
      {
         temp = _rawin[rawInNdx] - weights.at(netInNdx, rawInNdx);
         squared_euclidean_dist[netInNdx] += temp * temp;
      }

      spread[netInNdx] = exp (-weights.at(netInNdx, bias_ndx));
      netinv[netInNdx] = spread[netInNdx] * squared_euclidean_dist[netInNdx];
   }

   return netinv;
}

/**
 * Calculate the derivative of the net input with respect to the weights based on the raw
 * input std::vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
const Array2D<double>&
EuclideanDistLayer::calc_dNdW (const std::valarray<double> &_rawin)
{
   const Array2D<double>& weights = layer_weights.const_weights_ref;
   Array2D<double>& dNdW = layer_derivatives.dNdW;
   std::valarray<double>& netinv = layer_state.netinv;

   Array2D<double>::Dimensions wdim = weights.size();
   Array2D<double>::Dimensions ddim = dNdW.size();

   if (_rawin.size () + 1 != wdim.cols)
      throw std::invalid_argument ("external input vector size doesn't match column dimension in weight array");

   if (netinv.size () != wdim.rows)
      throw std::invalid_argument ("net input vector size doesn't match row dimension in weight array");

   if (ddim.rows != netinv.size () || ddim.cols != _rawin.size () + 1)
      throw std::invalid_argument ("dNdW array dimensionality doesn't match weight array");

   size_t spread_param_ndx = _rawin.size();
   for (size_t out_ndx = 0; out_ndx < const_layer_output_size_ref; out_ndx++)
   {
      // Calculate dNdW with respect to the spread parameter
      dNdW.at(out_ndx, spread_param_ndx) += -squared_euclidean_dist[out_ndx] * spread[out_ndx];

      // Calculate dNdW with respect to RBF kernel
      for (size_t in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
         dNdW.at(out_ndx, in_ndx) = -2.0 * spread[out_ndx] * (_rawin[in_ndx] - weights.at(out_ndx, in_ndx));

   }
   return dNdW;
}

/**
 * Calculate the derivative of the net input with respect to the raw input based on the raw
 * input valarray and weights specified in the argument list and writes it into the _dNdW argument.
 */
const Array2D<double>&
EuclideanDistLayer::calc_dNdI (const std::valarray<double> &_rawin)
{
   const Array2D<double>& weights = layer_weights.const_weights_ref;
   Array2D<double>& dNdI = layer_derivatives.dNdI;
   std::valarray<double>& netinv = layer_state.netinv;

   Array2D<double>::Dimensions wdim = weights.size();
   Array2D<double>::Dimensions ddim = dNdI.size();

   if (_rawin.size () + 1 != wdim.cols)
      throw std::invalid_argument ("input vector size doesn't match column dimension in weight array");

   if (ddim.rows != netinv.size () || ddim.cols != _rawin.size () + 1)
      throw std::invalid_argument ("dNdI array dimensionality doesn't match weight array");

   size_t spread_param_ndx = _rawin.size();
   for (size_t out_ndx = 0; out_ndx < netinv.size (); out_ndx++)
   {
      // TODO - probably not needed
      dNdI.at(out_ndx, spread_param_ndx) = 0;

      // Calculate dNdI with respect to input RBF kernel
      for (size_t in_ndx = 0; in_ndx < _rawin.size (); in_ndx++)
         dNdI.at(out_ndx, in_ndx) = 2.0 * spread[out_ndx] * (_rawin[in_ndx] - weights.at(out_ndx, in_ndx));
   }
   return dNdI;
}