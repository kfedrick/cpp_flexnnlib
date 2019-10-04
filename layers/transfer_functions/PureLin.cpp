//
// Created by kfedrick on 5/16/19.
//

#include "PureLin.h"

using flexnnet::PureLin;

const PureLin::Parameters PureLin::DEFAULT_PARAMS = {.gain=1.0};

PureLin::PureLin (size_t _sz, const std::string &_id, NetworkLayerType _type, const Parameters& _params)
   : NetSumLayer (_sz, _id, _type)
{
   layer_derivatives.dAdN.resize (_sz, _sz);
   set_params (_params);
}

PureLin::~PureLin ()
{
}

const std::valarray<double>&
PureLin::calc_layer_output (const std::valarray<double> &_rawin)
{
   layer_state.netinv = calc_netin (_rawin);
   layer_state.outputv = layer_state.netinv;

   return layer_state.outputv;
}

const flexnnet::Array2D<double>& PureLin::calc_dAdN (const std::valarray<double> &_out)
{
   Array2D<double>& dAdN = layer_derivatives.dAdN;

   dAdN = 0;
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      dAdN.at (i, i) = 1;

   return dAdN;
}

