//
// Created by kfedrick on 5/19/19.
//

#include "LogSig.h"

#include <cmath>

using flexnnet::Array2D;
using flexnnet::LogSig;

const LogSig::Parameters LogSig::DEFAULT_PARAMS = {.gain=1.0};

LogSig::LogSig(size_t _sz, const std::string& _name, NetworkLayerType _type, const Parameters& _params)
   : NetSumLayer(_sz, _name, _type)
{
   set_params(_params);
}

LogSig::~LogSig()
{
}

const std::valarray<double>& LogSig::calc_layer_output(const std::valarray<double>& _rawin)
{
   std::valarray<double>& netinv = layer_state.netinv;
   std::valarray<double>& outputv = layer_state.outputv;

   netinv = calc_netin(_rawin);

   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      outputv[i] = 1.0 / (1.0 + exp(-params.gain * netinv[i]));
}

const Array2D<double>& LogSig::calc_dAdN(const std::valarray<double>& _out)
{
   Array2D<double>& dAdN = layer_derivatives.dAdN;

   dAdN = 0;
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      dAdN.at(i, i) = params.gain * _out[i] * (1 - _out[i]);

   return dAdN;
}