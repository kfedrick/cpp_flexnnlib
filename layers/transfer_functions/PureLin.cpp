//
// Created by kfedrick on 5/16/19.
//

#include "PureLin.h"

#include <iostream>
using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::PureLin;

const PureLin::Parameters PureLin::DEFAULT_PARAMS = {.gain=1.0};

PureLin::PureLin(size_t _sz, const std::string& _id, const Parameters& _params)
   : NetSumLayer(_sz, _id)
{
   layer_derivatives.dy_dnet.resize(_sz, _sz);
   set_params(_params);
}

PureLin::PureLin(const PureLin& _purelin) : NetSumLayer(_purelin)
{
   copy(_purelin);
}

PureLin::~PureLin()
{
}

PureLin& PureLin::operator=(const PureLin& _purelin)
{
   copy(_purelin);
   return *this;
}

void PureLin::copy(const PureLin& _purelin)
{
   params = _purelin.params;
}

std::shared_ptr<BasicLayer> PureLin::clone(void) const
{
   std::shared_ptr<PureLin> clone = std::make_shared<PureLin>(PureLin(*this));
   return clone;
}

const std::valarray<double>&
PureLin::calc_layer_output(const std::valarray<double>& _rawin)
{
   layer_state.netinv = calc_netin(_rawin);
   layer_state.outputv = layer_state.netinv;

   return layer_state.outputv;
}

const Array2D<double>& PureLin::calc_dy_dnet(const std::valarray<double>& _out)
{
   Array2D<double>& dy_dnet = layer_derivatives.dy_dnet;

   dy_dnet = 0.0;
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      dy_dnet.at(i, i) = 1.0;

   return dy_dnet;
}

