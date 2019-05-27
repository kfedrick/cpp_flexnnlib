//
// Created by kfedrick on 5/9/19.
//

#ifndef FLEX_NEURALNET_LAYER_H_
#define FLEX_NEURALNET_LAYER_H_

#include "NetworkLayer.h"
#include "LayerSerializer.h"

namespace flexnnet
{
   template<class _TransferFunction>
   class Layer : public NetworkLayer, public _TransferFunction
   {

   public:
      static const Layer<_TransferFunction>& parse(const std::string& _json);

   public:
      Layer (unsigned int _sz = 0, const std::string &_name = "");
      ~Layer ();

      const std::string &name () const;

   protected:
      void resize_layer(unsigned int _layer_sz, unsigned int _rawin_sz);

   private:
      void bind_functions ();
   };

   template<class _TransFunc> Layer<_TransFunc>::Layer (unsigned int _sz, const std::string &_name)
      : NetworkLayer (_sz, _name), _TransFunc (_sz)
   {
         bind_functions();
   }

   template<class _TransFunc> Layer<_TransFunc>::~Layer ()
   {
   }

   template<class _TransFunc> const std::string &Layer<_TransFunc>::name () const
   {
      NetworkLayer::name ();
   }

   template<class _TransFunc> void Layer<_TransFunc>::bind_functions ()
   {
      using namespace std::placeholders;

      BasicLayer::resize_layer = std::bind (&Layer<_TransFunc>::resize_layer, this, _1, _2);

      BasicLayer::calc_netin = std::bind (&_TransFunc::calc_netin, this, _1, _2);
      BasicLayer::calc_dNdW = std::bind (&_TransFunc::calc_dNdW, this, _1, _2, _3);
      BasicLayer::calc_dNdI = std::bind (&_TransFunc::calc_dNdI, this, _1, _2, _3);

      BasicLayer::calc_layer_output = std::bind (&_TransFunc::calc_layer_output, this, _1, _2);
      BasicLayer::calc_dAdN = std::bind (&_TransFunc::calc_dAdN, this, _1, _2);

      BasicLayer::toJSONString = std::bind (&LayerSerializer< Layer<_TransFunc> >::toJSON, *this);
   }

   template<class _TransFunc> void Layer<_TransFunc>::resize_layer(unsigned int _layer_sz, unsigned int _rawin_sz)
   {
      // Save new layer input size
      layer_input_size = _rawin_sz;

      // Resize raw input vector cache
      layer_state.rawinv.resize(_rawin_sz);

      // Resize layer weights and derivatives
      layer_weights.resize(const_layer_output_size_ref, _rawin_sz);
      layer_derivatives.resize(const_layer_output_size_ref, const_layer_output_size_ref, _rawin_sz);

      // Resize transfer function data members
      _TransFunc::resize(_layer_sz, _rawin_sz);
   }


}

#endif //FLEXNEURALNET_LAYER_H_
