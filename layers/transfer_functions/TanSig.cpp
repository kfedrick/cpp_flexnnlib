//
// Created by kfedrick on 5/25/19.
//

#include "TanSig.h"
#include <cmath>

using flexnnet::Array2D;
using flexnnet::TanSig;

const TanSig::Parameters TanSig::DEFAULT_PARAMS = {.gain=1.0};

flexnnet::TanSig::TanSig(size_t _sz, const std::string& _name, NetworkLayerType _type, const Parameters& _params)
   : NetSumLayer(_sz, _name, _type)
{
   set_params(_params);
}

TanSig::~TanSig()
{
}

const std::valarray<double>& TanSig::calc_layer_output(const std::valarray<double>& _rawin)
{
   std::valarray<double>& netinv = layer_state.netinv;
   std::valarray<double>& outputv = layer_state.outputv;

   netinv = calc_netin(_rawin);
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      outputv[i] = 2.0 / (1.0 + exp(-2.0 * params.gain * (netinv[i]))) - 1.0;
}

const Array2D<double>& TanSig::calc_dAdN(const std::valarray<double>& _out)
{
   Array2D<double>& dAdN = layer_derivatives.dAdN;

   dAdN = 0;
   for (unsigned int i = 0; i < const_layer_output_size_ref; i++)
      dAdN.at(i, i) = params.gain * (1 - _out[i] * _out[i]);

   return dAdN;
}