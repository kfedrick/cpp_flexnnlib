//
// Created by kfedrick on 5/25/19.
//

#include "TanSig.h"
#include <cmath>

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::TanSig;

const TanSig::Parameters TanSig::DEFAULT_PARAMS = {.gain=1.0};

TanSig::TanSig(size_t _sz, const std::string& _name, const Parameters& _params)
   : NetSumLayer(_sz, _name)
{
   set_params(_params);
}

TanSig::TanSig(const TanSig& _tansig) : NetSumLayer(_tansig)
{
   copy(_tansig);
}

TanSig::~TanSig()
{
}

TanSig& TanSig::operator=(const TanSig& _tansig)
{
   copy(_tansig);
   return *this;
}

void TanSig::copy(const TanSig& _tansig)
{
   params = _tansig.params;
}

std::shared_ptr<BasicLayer> TanSig::clone(void) const
{
   std::shared_ptr<TanSig> clone = std::shared_ptr<TanSig>(new TanSig(*this));
   return clone;
}

const std::valarray<double>& TanSig::calc_layer_output(const std::valarray<double>& _rawin)
{
   std::valarray<double>& netinv = layer_state.netinv;
   std::valarray<double>& outputv = layer_state.outputv;

   netinv = calc_netin(_rawin);
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      outputv[i] = 2.0 / (1.0 + exp(-2.0 * params.gain * (netinv[i]))) - 1.0;
}

const Array2D<double>& TanSig::calc_dy_dnet(const std::valarray<double>& _out)
{
   Array2D<double>& dAdN = layer_derivatives.dy_dnet;

   dAdN = 0;
   for (unsigned int i = 0; i < const_layer_output_size_ref; i++)
      dAdN.at(i, i) = params.gain * (1 - _out[i] * _out[i]);

   return dAdN;
}