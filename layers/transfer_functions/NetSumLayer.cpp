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
 * Calculate the net input vectorize based on the raw input std::vector and weights specified in the
 * argument list and writes it into the _netin argument.
 */
void NetSumLayer::calc_netin(const std::valarray<double>& _rawinv, std::valarray<double>& _netinv)
{
   const Array2D<double>& lweights = weights().const_weights_ref;
   Array2D<double>::Dimensions wdim = lweights.size();

   size_t rawinv_sz = _rawinv.size();
   size_t netinv_sz = _netinv.size();

   if (rawinv_sz + 1 != wdim.cols)
      throw std::invalid_argument("input std::vector size doesn't match column dimension in weight array");

   if (netinv_sz != wdim.rows)
      throw std::invalid_argument("net input std::vector size doesn't match row dimension in weight array");

   size_t bias_ndx = rawinv_sz;
   for (size_t netin_ndx = 0; netin_ndx < netinv_sz; netin_ndx++)
   {
      // Initialize netin vectorize with bias, then add weighted sum of input vector
      _netinv[netin_ndx] = lweights.at(netin_ndx, bias_ndx);
      for (size_t rawin_ndx = 0; rawin_ndx < _rawinv.size(); rawin_ndx++)
         _netinv[netin_ndx] += _rawinv[rawin_ndx] * lweights.at(netin_ndx, rawin_ndx);
   }
}

/**
 * Calculate the derivative of the net input with respect to the weights based on the raw
 * input vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
void NetSumLayer::calc_dnet_dw(const LayerState& _lstate, Array2D<double>& _dnetdw)
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
      throw std::invalid_argument("external input valarray size doesn't match column dimension in weight array");

   if (netinv_sz != wdim.rows)
      throw std::invalid_argument("net input vector size doesn't match row dimension in weight array");

   if (ddim.rows != netinv_sz || ddim.cols != rawinv_sz + 1)
      throw std::invalid_argument("dnet_dw array dimensionality doesn't match weight array");

   for (size_t netin_ndx = 0; netin_ndx < netinv_sz; netin_ndx++)
   {
      _dnetdw.at(netin_ndx, bias_ndx) = 1;
      for (size_t in_ndx = 0; in_ndx < rawinv_sz; in_ndx++)
         _dnetdw.at(netin_ndx, in_ndx) = rawinv[in_ndx];
   }
}

/**
 * Calculate the derivative of the net input with respect to the raw input based on the raw
 * input vector and weights specified in the argument list and writes it into the _dNdW argument.
 */
void NetSumLayer::calc_dnet_dx(const LayerState& _lstate, Array2D<double>& _dnetdx)
{
   const Array2D<double>& lweights = weights().const_weights_ref;

   const std::valarray<double>& netinv = _lstate.netinv;
   const std::valarray<double>& rawinv = _lstate.rawinv;

   Array2D<double>::Dimensions wdim = lweights.size();
   Array2D<double>::Dimensions ddim = _dnetdx.size();

   size_t netinv_sz = netinv.size();
   size_t rawinv_sz = rawinv.size();

   if (rawinv.size() + 1 != wdim.cols)
      throw std::invalid_argument("input vector size doesn't match column dimension in weight array");

   if (ddim.rows != netinv_sz || ddim.cols != rawinv_sz)
      throw std::invalid_argument("dnet_dx array dimensionality doesn't match weight array");


   for (size_t netin_ndx = 0; netin_ndx < netinv_sz; netin_ndx++)
      for (size_t in_ndx = 0; in_ndx < rawinv_sz; in_ndx++)
         _dnetdx.at(netin_ndx, in_ndx) = lweights.at(netin_ndx, in_ndx);
}