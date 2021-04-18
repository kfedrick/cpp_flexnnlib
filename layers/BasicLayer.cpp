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
   layer_weights = _basic_layer.layer_weights;
}

void BasicLayer::activate(const std::valarray<double>& _rawinv, LayerState& _lstate)
{
/*   std::cout << "BaseLayer::rawinv " << name() << "\n";
   for (auto v : _lstate.rawinv)
      std::cout << v << " ";
   std::cout << "\n";*/

   _lstate.rawinv = _rawinv;
   calc_netin(_rawinv, _lstate.netinv);

/*   std::cout << "BaseLayer::netinv " << name() << "\n";
   for (auto v : _lstate.netinv)
      std::cout << v << " ";
   std::cout << "\n";*/

   calc_layer_output(_lstate.netinv, _lstate.outputv);
}

void BasicLayer::backprop(const std::valarray<double>& _dEdy, LayerState& _state)
{
   // Cache external error (instantaneous partial derivative of
   // the sample error wrt the layer output, y).
   _state.dE_dy = _dEdy;

   /*
    * Back-propagate error through transfer function.
    */
   calc_dy_dnet(_state.outputv, _state.dy_dnet);

   _state.dE_dnet = 0;
   for (unsigned int netin_ndx = 0; netin_ndx < layer_output_size; netin_ndx++)
      for (unsigned int out_ndx = 0; out_ndx < layer_output_size; out_ndx++)
         _state.dE_dnet[netin_ndx] += _dEdy[out_ndx] * _state.dy_dnet.at(out_ndx, netin_ndx);

   /*
    * Calculate instantaneous partial derivative of error wrt weights
    * so that we can calculate the weight updates.
    */
   calc_dnet_dw(_state, _state.dnet_dw);

   _state.dE_dw = 0;
   for (unsigned int in_ndx = 0; in_ndx < layer_input_size+1; in_ndx++)
      for (unsigned int netin_ndx = 0; netin_ndx < layer_output_size; netin_ndx++)
         _state.dE_dw.at(netin_ndx, in_ndx) = _state.dE_dnet[netin_ndx] * _state.dnet_dw.at(netin_ndx, in_ndx);

   /*
    * Back-propagate net input error through net input function to
    * calculate the input error, dEdx.
    */
   calc_dnet_dx(_state, _state.dnet_dx);

   double dnet_dx_sum = 0;
   _state.dE_dx = 0;
   for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
      for (unsigned int netin_ndx = 0; netin_ndx < layer_output_size; netin_ndx++)
      {
         dnet_dx_sum += fabs(_state.dnet_dx.at(netin_ndx, in_ndx));
         _state.dE_dx[in_ndx] +=
            _state.dE_dnet[netin_ndx] * _state.dnet_dx.at(netin_ndx, in_ndx);
      }

   // If the absolute sum of dnet_dx is very close to zero then add a
   // tiny bit of noise to the input error vector to help break the
   // symmetry.
   if (dnet_dx_sum < 1e-9)
   {
      std::normal_distribution<double> normal_dist(0, 1e-6);
      for (auto& v : _state.dE_dx)
         v += normal_dist(rand_engine);
   }
}