//
// Created by kfedrick on 5/8/19.
//

#include <iostream>
#include <valarray>
#include "BasicLayer.h"

using flexnnet::BasicLayer;

BasicLayer::BasicLayer(size_t _sz, const std::string& _name)
   : NamedObject(_name), layer_output_size(_sz), layer_input_size(0)
{
   layer_state.outputv.resize(_sz);
   layer_state.netinv.resize(_sz);
   layer_state.netin_errorv.resize(_sz);
   layer_state.external_errorv.resize(_sz);

   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);
}

BasicLayer::BasicLayer(const BasicLayer& _basic_layer) : NamedObject(_basic_layer.name()), layer_output_size(_basic_layer.layer_output_size)
{
   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);

   copy(_basic_layer);
}

BasicLayer::~BasicLayer()
{
}

void BasicLayer::copy(const BasicLayer& _basic_layer)
{
   layer_input_size = _basic_layer.layer_input_size;
   layer_state = _basic_layer.layer_state;
   layer_derivatives = _basic_layer.layer_derivatives;
   layer_weights = _basic_layer.layer_weights;
}

const std::valarray<double>& BasicLayer::activate(const std::valarray<double>& _rawin)
{
   layer_state.rawinv = _rawin;
   calc_layer_output(_rawin);

   layer_derivatives.stale();

   //calc_dy_dnet(layer_state.outputv);
   //calc_dnet_dw(_rawin);
   //calc_dnet_dx(_rawin);

   return layer_state.outputv;
}

const std::valarray<double>& BasicLayer::backprop(const std::valarray<double>& _errorv)
{
   std::valarray<double>& netin_errorv = layer_state.netin_errorv;
   std::valarray<double>& input_errorv = layer_state.input_errorv;

   // Save the external layer error.
   layer_state.external_errorv = _errorv;

   /*
    * Back-propagate error to calculate error vector for net input.
    */
   const Array2D<double>& curr_dy_dnet = calc_dy_dnet(layer_state.outputv);

   netin_errorv = 0;
   for (unsigned int netin_ndx = 0; netin_ndx < layer_output_size; netin_ndx++)
      for (unsigned int out_ndx = 0; out_ndx < layer_output_size; out_ndx++)
         netin_errorv[netin_ndx] += _errorv[out_ndx] * curr_dy_dnet.at(out_ndx, netin_ndx);

   /*
    * Calculate instantaneous partial derivative of error wrt weights
    */
   const Array2D<double>& curr_dnet_dw = calc_dnet_dw(layer_state.rawinv);

   Array2D<double>& curr_dE_dw = layer_derivatives.dE_dw;
   curr_dE_dw = 0;
   for (unsigned int in_ndx = 0; in_ndx < layer_input_size+1; in_ndx++)
      for (unsigned int netin_ndx = 0; netin_ndx < layer_output_size; netin_ndx++)
         curr_dE_dw.at(netin_ndx, in_ndx) = netin_errorv[netin_ndx] * curr_dnet_dw.at(netin_ndx, in_ndx);

   /*
    * Back-propagate net input error through netsum function to
    * calculate the input error.
    */
   const Array2D<double>& curr_dnet_dx = calc_dnet_dx(layer_state.rawinv);

   double dnet_dx_sum = 0;
   input_errorv = 0;
   for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
      for (unsigned int netin_ndx = 0; netin_ndx < layer_output_size; netin_ndx++)
      {
         dnet_dx_sum += fabs(curr_dnet_dx.at(netin_ndx, in_ndx));
         input_errorv[in_ndx] +=
            netin_errorv[netin_ndx] * curr_dnet_dx.at(netin_ndx, in_ndx);
      }

   // If the absolute sum of dnet_dx is very close to zero then add a
   // tiny bit of noise to the input error vector to help break the
   // symmetry.
   if (dnet_dx_sum < 1e-7)
   {
      std::normal_distribution<double> normal_dist(0, 1e-7);
      for (auto& v : layer_state.input_errorv)
         v = normal_dist(rand_engine);
   }

   return layer_state.input_errorv;
}