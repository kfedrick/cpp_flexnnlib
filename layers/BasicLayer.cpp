//
// Created by kfedrick on 5/8/19.
//

#include <iostream>
#include "BasicLayer.h"

using flexnnet::BasicLayer;

BasicLayer::BasicLayer (size_t _sz, const std::string &_name, NetworkLayerType _type)
   : NamedObject (_name), layer_output_size (_sz), layer_input_size (0), network_layer_type (_type)
{
   layer_state.outputv.resize (_sz);
   layer_state.netinv.resize (_sz);
}

BasicLayer::~BasicLayer ()
{
}

const std::valarray<double> &BasicLayer::activate (const std::valarray<double> &_rawin)
{
   layer_state.rawinv = _rawin;

   Array2D<double>::Dimensions dim = layer_weights.const_weights_ref.size ();

   calc_layer_output (_rawin);

   layer_derivatives.stale ();

   calc_dAdN (layer_state.outputv);
   calc_dNdW (_rawin);
   calc_dNdI (_rawin);

   return layer_state.outputv;
}

const std::valarray<double> &BasicLayer::operator() () const
{
   return layer_state.outputv;
}

const std::valarray<double> &BasicLayer::accumulate_error (const std::valarray<double> &_errorv)
{
   for (size_t ndx = 0; ndx < layer_state.backprop_errorv.size (); ndx++)
   {
      layer_state.backprop_errorv[ndx] += _errorv[ndx];
   }
}