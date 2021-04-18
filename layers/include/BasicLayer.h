//
// Created by kfedrick on 5/8/19.
//

#ifndef FLEX_NEURALNET_BASICLAYER_H_
#define FLEX_NEURALNET_BASICLAYER_H_

#include <set>
#include <unordered_set>
#include <sstream>
#include <functional>
#include <memory>

#include "NamedObject.h"
#include "LayerWeights.h"
#include "LayerState.h"

namespace flexnnet
{
   class BasicLayer : public NamedObject
   {

   protected:

      /* ********************************************************************
       * Constructors, destructors
       */
      BasicLayer(size_t _sz, const std::string& _name);
      BasicLayer(const BasicLayer& _basic_layer);

   public:
      virtual std::shared_ptr<BasicLayer> clone(void) const = 0;
      virtual ~BasicLayer();

   public:

      // Return length of basic_layer output valarray
      size_t size() const;
      virtual size_t input_size() const;
      virtual const LayerWeights& weights(void) const;
      virtual LayerWeights& weights(void);

   public:
      /* ********************************************************************
       * Public basic_layer operational methods
       */

      /**
       * Calculate the value of the basic_layer neuron vector based on the specified
       * raw input vectors.
       * @param inputVec
       * @return
       */
      virtual void activate(const std::valarray<double>& _rawinv, LayerState& _lstate);

      /**
       * Calculates the net input error vector and input error vector given
       * the layer output error vector.
       *
       * @param _errorv
       * @return
       */
      virtual void backprop(const std::valarray<double>& _dEdy, LayerState& _lstate);

      virtual void resize_input(size_t _rawin_sz);

   protected:

      virtual void calc_layer_output(const std::valarray<double>& _netin, std::valarray<double>& _layerval) = 0;

      /**
       * Calculate the net input value based on the raw input vector and weights specified in the
       * argument list and writes it into the _netin argument.
       */
      virtual void calc_netin(const std::valarray<double>& _rawin, std::valarray<double>& _netin) = 0;

      /*
       * Calculate and return the derivative of the basic_layer output with respect to
       * the net input for the most recent activation.
       */
      virtual void calc_dy_dnet(const std::valarray<double>& _outv, Array2D<double>& _dydnet) = 0;

      /**
       * Calculate the derivative of the net input with respect to the weights based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      virtual void calc_dnet_dw(const LayerState& _lstate, Array2D<double>& _dnetdw) = 0;

      /**
       * Calculate the derivative of the net input with respect to the raw input based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      virtual void calc_dnet_dx(const LayerState& _lstate, Array2D<double>& _dnetdx) = 0;

      mutable std::mt19937_64 rand_engine;

   private:
      void copy(const BasicLayer& _basic_layer);

   public:
      const size_t& const_layer_output_size_ref = layer_output_size;
      const size_t& const_layer_input_size_ref = layer_input_size;

   private:

      /* ********************************************************************
       * Private basic_layer configuration and provisioning data members
       */
      size_t layer_output_size;
      size_t layer_input_size;

      LayerWeights layer_weights;
   };

   inline size_t BasicLayer::size() const
   {
      return layer_output_size;
   }

   inline size_t BasicLayer::input_size() const
   {
      return layer_input_size;
   }

   inline void BasicLayer::resize_input(size_t _rawin_sz)
   {
      layer_input_size = _rawin_sz;
      layer_weights.resize(layer_output_size, layer_input_size);
   }

   inline
   LayerWeights& BasicLayer::weights(void)
   {
      return layer_weights;
   }

   inline
   const LayerWeights& BasicLayer::weights(void) const
   {
      return layer_weights;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_BASICLAYER_H_ */