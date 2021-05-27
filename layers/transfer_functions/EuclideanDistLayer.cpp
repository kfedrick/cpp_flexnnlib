//
// Created by kfedrick on 6/10/19.
//

#include "EuclideanDistLayer.h"

using flexnnet::Array2D;
using flexnnet::EuclideanDistLayer;

EuclideanDistLayer::EuclideanDistLayer(size_t _sz, const std::string& _name)
   : BasicLayer(_sz, _name)
{
   squared_euclidean_dist.resize(_sz);
   spread.resize(_sz);
}

EuclideanDistLayer::~EuclideanDistLayer()
{
}

/**
 * Calculate the net input vectorize based on the raw input std::vector and weights specified in the
 * argument list and writes it into the _netin argument.
 */
void
EuclideanDistLayer::calc_netin(const std::valarray<double>& _rawinv, std::valarray<double>& _netinv)
{
   static double temp;

   const Array2D<double>& lweights = weights().const_weights_ref;
   Array2D<double>::Dimensions wdim = lweights.size();

   size_t rawinv_sz = _rawinv.size();
   size_t netinv_sz = _netinv.size();

   if (rawinv_sz + 1 != wdim.cols)
      throw std::invalid_argument("input vector size doesn't match column dimension in weight array");

   if (netinv_sz != wdim.rows)
      throw std::invalid_argument("net input vector size doesn't match row dimension in weight array");

   /*
    * Calculate actvity to the netInNdx neuron as the euclidean distance
    * from the input vector to the vector specified by the weights vector
    * to the netInNdx neuron. Use the bias as a spread parameter.
    */
   size_t bias_ndx = _rawinv.size();
   for (size_t netin_ndx = 0; netin_ndx < const_layer_output_size_ref; netin_ndx++)
   {
      squared_euclidean_dist[netin_ndx] = 0;
      for (unsigned int rawin_ndx = 0; rawin_ndx < const_layer_input_size_ref;
           rawin_ndx++)
      {
         temp = _rawinv[rawin_ndx] - lweights.at(netin_ndx, rawin_ndx);
         squared_euclidean_dist[netin_ndx] += temp * temp;
      }

      spread[netin_ndx] = exp(-lweights.at(netin_ndx, bias_ndx));
      _netinv[netin_ndx] = spread[netin_ndx] * squared_euclidean_dist[netin_ndx];
   }
}

/**
 * Calculate the derivative of the net input with respect to the weights based on the raw
 * input std::vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
void
EuclideanDistLayer::calc_dnet_dw(const LayerState& _lstate, Array2D<double>& _dnetdw)
{
   const Array2D<double>& lweights = weights().const_weights_ref;

   const std::valarray<double>& rawinv = _lstate.rawinv;
   const std::valarray<double>& netinv = _lstate.netinv;

   Array2D<double>::Dimensions wdim = lweights.size();
   Array2D<double>::Dimensions ddim = _dnetdw.size();

   size_t bias_ndx = rawinv.size();
   size_t netinv_sz = netinv.size();
   size_t rawinv_sz = rawinv.size();

   if (rawinv_sz + 1 != wdim.cols)
      throw std::invalid_argument("external input vector size doesn't match column dimension in weight array");

   if (netinv_sz != wdim.rows)
      throw std::invalid_argument("net input vector size doesn't match row dimension in weight array");

   if (ddim.rows != netinv_sz || ddim.cols != rawinv_sz + 1)
      throw std::invalid_argument("dnet_dw array dimensionality doesn't match weight array");

   size_t spread_param_ndx = rawinv_sz;
   for (size_t out_ndx = 0; out_ndx < const_layer_output_size_ref; out_ndx++)
   {
      // Calculate dnet_dw with respect to the spread parameter
      _dnetdw.at(out_ndx, spread_param_ndx) =
         -squared_euclidean_dist[out_ndx] * spread[out_ndx];

      // Calculate dnet_dw with respect to RBF kernel
      for (size_t in_ndx = 0; in_ndx < const_layer_input_size_ref; in_ndx++)
         _dnetdw.at(out_ndx, in_ndx) =
            -2.0 * spread[out_ndx] * (rawinv[in_ndx] - lweights.at(out_ndx, in_ndx));

   }
}

/**
 * Calculate the derivative of the net input with respect to the raw input based on the raw
 * input valarray and weights specified in the argument list and writes it into the _dNdW argument.
 */
void
EuclideanDistLayer::calc_dnet_dx(const LayerState& _lstate, Array2D<double>& _dnetdx)
{
   const Array2D<double>& lweights = weights().const_weights_ref;

   const std::valarray<double>& netinv = _lstate.netinv;
   const std::valarray<double>& rawinv = _lstate.rawinv;

   Array2D<double>::Dimensions wdim = lweights.size();
   Array2D<double>::Dimensions ddim = _dnetdx.size();

   size_t netinv_sz = netinv.size();
   size_t rawinv_sz = rawinv.size();

   if (rawinv_sz + 1 != wdim.cols)
      throw std::invalid_argument("input vector size doesn't match column dimension in weight array");

   if (ddim.rows != netinv_sz || ddim.cols != rawinv_sz)
      throw std::invalid_argument("dnet_dx array dimensionality doesn't match weight array");

   for (size_t out_ndx = 0; out_ndx < netinv_sz; out_ndx++)
   {
      // Calculate dnet_dx with respect to input RBF kernel
      for (size_t in_ndx = 0; in_ndx < rawinv_sz; in_ndx++)
         _dnetdx.at(out_ndx, in_ndx) =
            2.0 * spread[out_ndx] * (rawinv[in_ndx] - lweights.at(out_ndx, in_ndx));
   }
}