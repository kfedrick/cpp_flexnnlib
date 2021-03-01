//
// Created by kfedrick on 5/8/19.
//

#include <iostream>
#include "BasicLayer.h"

using flexnnet::BasicLayer;

BasicLayer::BasicLayer(size_t _sz, const std::string& _name)
   : NamedObject(_name), layer_output_size(_sz), layer_input_size(0)
{
   layer_state.outputv.resize(_sz);
   layer_state.netinv.resize(_sz);
}

BasicLayer::BasicLayer(const BasicLayer& _basic_layer) : NamedObject(_basic_layer.name()), layer_output_size(_basic_layer.layer_output_size)
{
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

   calc_dAdN(layer_state.outputv);
   calc_dNdW(_rawin);
   calc_dNdI(_rawin);

   return layer_state.outputv;
}

const std::valarray<double>& BasicLayer::operator()() const
{
   return layer_state.outputv;
}

const std::valarray<double>& BasicLayer::accumulate_error(const std::valarray<double>& _errorv)
{
   for (size_t ndx = 0; ndx < layer_state.backprop_errorv.size(); ndx++)
   {
      layer_state.backprop_errorv[ndx] += _errorv[ndx];
   }
}