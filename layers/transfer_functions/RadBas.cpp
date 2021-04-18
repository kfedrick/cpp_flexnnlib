//
// Created by kfedrick on 5/24/19.
//

#include "RadBas.h"

#include <cmath>

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::RadBas;

const RadBas::Parameters RadBas::DEFAULT_PARAMS = {.rescaled_flag=false};

RadBas::RadBas(size_t _sz, const std::string& _name, const Parameters& _params)
   : EuclideanDistLayer(_sz, _name)
{
   set_params(_params);
}

RadBas::RadBas(const RadBas& _radbas) : EuclideanDistLayer(_radbas)
{
   copy(_radbas);
}

RadBas::~RadBas()
{
}

RadBas& RadBas::operator=(const RadBas& _radbas)
{
   copy(_radbas);
   return *this;
}

void RadBas::copy(const RadBas& _radbas)
{
   lower_bound = _radbas.lower_bound;
   output_range = _radbas.output_range;

   params = _radbas.params;
}

std::shared_ptr<BasicLayer> RadBas::clone(void) const
{
   std::shared_ptr<RadBas> clone = std::shared_ptr<RadBas>(new RadBas(*this));
   return clone;
}

void RadBas::calc_layer_output(const std::valarray<double>& _netinv, std::valarray<double>& _layerval)
{
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      _layerval[i] = output_range * exp(-_netinv[i]) + lower_bound;
}

void RadBas::calc_dy_dnet(const std::valarray<double>& _outv, Array2D<double>& _dydnet)
{
   _dydnet = 0;
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      _dydnet.at(i, i) = -output_range * _outv[i];
}