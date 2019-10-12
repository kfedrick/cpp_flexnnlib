//
// Created by kfedrick on 5/27/19.
//

#include "SoftMax.h"
#include <cmath>

using flexnnet::Array2D;
using flexnnet::SoftMax;

const SoftMax::Parameters SoftMax::DEFAULT_PARAMS = {.gain=1.0, .rescaled_flag=false};

flexnnet::SoftMax::SoftMax(size_t _sz, const std::string& _name, NetworkLayerType _type, const Parameters& _params)
   : NetSumLayer(_sz, _name, _type)
{
   set_params(_params);
   exp_netin.resize(_sz);
}

flexnnet::SoftMax::~SoftMax()
{
}

const std::valarray<double>& flexnnet::SoftMax::calc_layer_output(const std::valarray<double>& _rawin)
{
   double sum_exp = 0;

   std::valarray<double>& netinv = layer_state.netinv;
   std::valarray<double>& outputv = layer_state.outputv;

   netinv = calc_netin(_rawin);


   // Calculate the initial exp of the input values and accumulate the summation
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
   {
      exp_netin[i] = exp(params.gain * netinv[i]);
      sum_exp += exp_netin[i];
   }

   // Normalize the transfer vector by dividing by the sum of activity
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      outputv[i] = output_range * exp_netin[i] / sum_exp + lower_bound;
}

const Array2D<double>& SoftMax::calc_dAdN(const std::valarray<double>& _out)
{
   Array2D<double>& dAdN = layer_derivatives.dAdN;

   dAdN = 0;

   for (size_t netin_ndx = 0; netin_ndx < const_layer_output_size_ref; netin_ndx++)
   {
      for (size_t trans_ndx = 0; trans_ndx < const_layer_output_size_ref; trans_ndx++)
         dAdN.at(trans_ndx, netin_ndx) =
            (trans_ndx == netin_ndx) ?
            output_range * params.gain * _out[trans_ndx] * (1 - _out[trans_ndx]) :
            output_range * params.gain * -_out[trans_ndx] * _out[netin_ndx];
   }

   return dAdN;
}