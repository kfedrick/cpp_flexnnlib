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
   std::cout << "clone size = " << clone->size() << "x" << clone->input_size() << "\n";
   return clone;
}

void
PureLin::calc_layer_output(const std::valarray<double>& _netin, std::valarray<double>& _layerval)
{
   _layerval = _netin;
}

void PureLin::calc_dy_dnet(const std::valarray<double>& _outv, Array2D<double>& _dydnet)
{
   _dydnet = 0.0;
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      _dydnet.at(i, i) = 1.0;
}

