//
// Created by kfedrick on 5/27/19.
//

#include "SoftMax.h"
#include <cmath>

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::SoftMax;

const SoftMax::Parameters SoftMax::DEFAULT_PARAMS = {.gain=1.0, .rescaled_flag=false};

flexnnet::SoftMax::SoftMax(size_t _sz, const std::string& _name, const Parameters& _params)
   : NetSumLayer(_sz, _name)
{
   set_params(_params);
   exp_netin.resize(_sz);
}


SoftMax::SoftMax(const SoftMax& _softmax) : NetSumLayer(_softmax)
{
   copy(_softmax);
}

flexnnet::SoftMax::~SoftMax()
{
}

SoftMax& SoftMax::operator=(const SoftMax& _softmax)
{
   copy(_softmax);
   return *this;
}

void SoftMax::copy(const SoftMax& _softmax)
{
   params = _softmax.params;

   lower_bound = _softmax.lower_bound;
   output_range = _softmax.output_range;
   exp_netin = _softmax.exp_netin;
}

std::shared_ptr<BasicLayer> SoftMax::clone(void) const
{
   std::shared_ptr<SoftMax> clone = std::shared_ptr<SoftMax>(new SoftMax(*this));
   return clone;
}

void flexnnet::SoftMax::calc_layer_output(const std::valarray<double>& _netin, std::valarray<double>& _layerval)
{
   double sum_exp = 0;

   // Calculate the initial exp of the input values and accumulate the summation
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
   {
      exp_netin[i] = exp(params.gain * _netin[i]);
      sum_exp += exp_netin[i];
   }

   // Normalize the transfer vector by dividing by the sum of activity
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      _layerval[i] = output_range * exp_netin[i] / sum_exp + lower_bound;
}

void SoftMax::calc_dy_dnet(const std::valarray<double>& _outv, Array2D<double>& _dydnet)
{
   _dydnet = 0;
   for (size_t netin_ndx = 0; netin_ndx < const_layer_output_size_ref; netin_ndx++)
   {
      for (size_t out_ndx = 0; out_ndx < const_layer_output_size_ref; out_ndx++)
         _dydnet.at(out_ndx, netin_ndx) =
            (out_ndx == netin_ndx) ?
            output_range * params.gain * _outv[out_ndx] * (1 - _outv[out_ndx]) :
            output_range * params.gain * -_outv[out_ndx] * _outv[netin_ndx];
   }
}