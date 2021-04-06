//
// Created by kfedrick on 6/10/19.
//

#include <cstddef>
#include <string>

#include "NetSumLayer.h"

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::NetSumLayer;

NetSumLayer::NetSumLayer(size_t _sz, const std::string& _name) : BasicLayer(_sz, _name)
{
}

NetSumLayer::NetSumLayer(const NetSumLayer& _netsum) : BasicLayer(_netsum)
{
}

NetSumLayer::~NetSumLayer()
{
}

/**
 * Calculate the net input value based on the raw input std::vector and weights specified in the
 * argument list and writes it into the _netin argument.
 */
const std::valarray<double>& NetSumLayer::calc_netin(const std::valarray<double>& _rawin)
{
   const Array2D<double>& weights = layer_weights.const_weights_ref;
   Array2D<double>::Dimensions wdim = weights.size();

   if (_rawin.size() + 1 != wdim.cols)
      throw std::invalid_argument("input std::vector size doesn't match column dimension in weight array");

   if (layer_state.netinv.size() != wdim.rows)
      throw std::invalid_argument("net input std::vector size doesn't match row dimension in weight array");

   size_t bias_ndx = _rawin.size();
   std::valarray<double>& netin = layer_state.netinv;
   for (size_t netInNdx = 0; netInNdx < const_layer_output_size_ref; netInNdx++)
   {
      // Initialize netin value with bias, then add weighted sum of input vector
      netin[netInNdx] = weights.at(netInNdx, bias_ndx);
      for (size_t rawInNdx = 0; rawInNdx < _rawin.size(); rawInNdx++)
         netin[netInNdx] += _rawin[rawInNdx] * weights.at(netInNdx, rawInNdx);
   }

   return netin;
}

/**
 * Calculate the derivative of the net input with respect to the weights based on the raw
 * input vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
const Array2D<double>& NetSumLayer::calc_dnet_dw(const std::valarray<double>& _rawin)
{
   const Array2D<double>& weights = layer_weights.const_weights_ref;
   Array2D<double>& dnet_dw = layer_derivatives.dnet_dw;
   std::valarray<double>& netinv = layer_state.netinv;

   Array2D<double>::Dimensions wdim = weights.size();
   Array2D<double>::Dimensions ddim = dnet_dw.size();

   if (_rawin.size() + 1 != wdim.cols)
      throw std::invalid_argument("external input valarray size doesn't match column dimension in weight array");

   if (netinv.size() != wdim.rows)
      throw std::invalid_argument("net input vector size doesn't match row dimension in weight array");

   if (ddim.rows != netinv.size() || ddim.cols != _rawin.size() + 1)
      throw std::invalid_argument("dnet_dw array dimensionality doesn't match weight array");

   size_t bias_ndx = _rawin.size();
   std::valarray<double>& netin = layer_state.netinv;
   for (size_t out_ndx = 0; out_ndx < netin.size(); out_ndx++)
   {
      dnet_dw.at(out_ndx, bias_ndx) = 1;
      for (size_t in_ndx = 0; in_ndx < _rawin.size(); in_ndx++)
         dnet_dw.at(out_ndx, in_ndx) = _rawin[in_ndx];
   }
   return dnet_dw;
}

/**
 * Calculate the derivative of the net input with respect to the raw input based on the raw
 * input vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
const Array2D<double>& NetSumLayer::calc_dnet_dx(const std::valarray<double>& _rawin)
{
   const Array2D<double>& weights = layer_weights.const_weights_ref;

   Array2D<double>::Dimensions wdim = weights.size();
   Array2D<double>::Dimensions ddim = layer_derivatives.dnet_dx.size();

   if (_rawin.size() + 1 != wdim.cols)
      throw std::invalid_argument("input vector size doesn't match column dimension in weight array");

   if (ddim.rows != layer_state.netinv.size() || ddim.cols != _rawin.size())
      throw std::invalid_argument("dnet_dx array dimensionality doesn't match weight array");

   const std::valarray<double>& netin = layer_state.netinv;
   for (size_t out_ndx = 0; out_ndx < netin.size(); out_ndx++)
      for (size_t in_ndx = 0; in_ndx < _rawin.size(); in_ndx++)
         layer_derivatives.dnet_dx.at(out_ndx, in_ndx) = weights.at(out_ndx, in_ndx);

   return layer_derivatives.dnet_dx;
}