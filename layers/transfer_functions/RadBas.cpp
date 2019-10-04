//
// Created by kfedrick on 5/24/19.
//

#include "RadBas.h"

#include <cmath>

using flexnnet::Array2D;
using flexnnet::RadBas;

const RadBas::Parameters RadBas::DEFAULT_PARAMS = {.rescaled_flag=false};

RadBas::RadBas(size_t _sz, const std::string &_name, NetworkLayerType _type, const Parameters& _params) : EuclideanDistLayer(_sz, _name, _type)
{
   set_params (_params);
}

RadBas::~RadBas()
{
}

const std::valarray<double>& RadBas::calc_layer_output (const std::valarray<double>& _rawin)
{
   std::valarray<double>& netinv = layer_state.netinv;
   std::valarray<double>& outputv = layer_state.outputv;

   netinv = calc_netin (_rawin);

   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      outputv[i] = output_range * exp (-netinv[i]) + lower_bound;
}

const Array2D<double>& RadBas::calc_dAdN(const std::valarray<double>& _out)
{
   Array2D<double>& dAdN = layer_derivatives.dAdN;

   dAdN = 0;
   for (size_t i = 0; i < _out.size (); i++)
      dAdN.at(i, i) = -output_range * _out[i];

   return dAdN;
}