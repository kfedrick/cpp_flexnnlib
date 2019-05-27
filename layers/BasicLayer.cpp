//
// Created by kfedrick on 5/8/19.
//

#include <iostream>
#include "BasicLayer.h"

namespace flexnnet
{
   BasicLayer::BasicLayer (unsigned int _sz, const std::string &_name )
      : NamedObject (_name), layer_output_size (_sz)
   {
      layer_input_size = 0;

      layer_state.outputv.resize(_sz);
      layer_state.netinv.resize(_sz);
      layer_weights.resize(layer_output_size, layer_input_size);
   }

   BasicLayer::~BasicLayer ()
   {

   }

   void BasicLayer::resize_input_vector (unsigned int _sz)
   {
      layer_input_size = _sz;
      resize_layer(layer_output_size, layer_input_size);
/*
      layer_state.rawinv.resize(layer_input_size);

      layer_weights.resize(layer_output_size, layer_input_size);
      layer_derivatives.resize(layer_output_size, layer_output_size, layer_input_size);
      */
   }


   const std::vector<double> &BasicLayer::activate (const std::vector<double> &_rawin)
   {
      layer_state.rawinv = _rawin;

      std::cout << "_rawin size " << _rawin.size() << std::endl;
      std::cout << "_netin size " << layer_state.netinv.size() << std::endl;

      std::cout << "weight size (" << layer_weights.const_weights_ref.rowDim() << "," << layer_weights.const_weights_ref.colDim() << ")" << std::endl;
      layer_state.netinv = calc_netin(_rawin, layer_weights.const_weights_ref);

      printf("rawin\n");
      for (unsigned int i=0; i<_rawin.size(); i++)
         printf("%7.5f ", _rawin[i]);
      printf("\n\n");

      printf("netin\n");
      for (unsigned int i=0; i<layer_state.netinv.size(); i++)
         printf("%7.5f ", layer_state.netinv[i]);
      printf("\n\n");

      calc_layer_output(layer_state.outputv, layer_state.netinv);

      layer_derivatives.stale();

      return layer_state.outputv;
   }

   const std::vector<double> &BasicLayer::operator() () const
   {
      return layer_state.outputv;
   }

   const std::vector<double>& BasicLayer::accumulate_error(const std::vector<double>& _errorv)
   {
      for (unsigned int ndx = 0; ndx < layer_state.backprop_errorv.size (); ndx++)
      {
         layer_state.backprop_errorv.at (ndx) += _errorv.at (ndx);
      }
   }


}
