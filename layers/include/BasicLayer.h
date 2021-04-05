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
#include "LayerDerivatives.h"
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

      const LayerState& state(void) const;

      const Array2D<double>& get_dy_dnet(void) const;
      const Array2D<double>& get_dnet_dw(void) const;
      const Array2D<double>& get_dnet_dx(void) const;
      const Array2D<double>& get_dE_dw(void) const;

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
      virtual const std::valarray<double>& activate(const std::valarray<double>& _inputv);

      /**
       * Calculates the net input error vector and input error vector given
       * the layer output error vector.
       *
       * @param _errorv
       * @return
       */
      virtual const std::valarray<double>& backprop(const std::valarray<double>& _errorv);

      /*
       * Return the current value of the network basic_layer as a std::valarray
       */
      virtual const std::valarray<double>& operator()() const;

      virtual const std::valarray<double>& input_error(void) const;

      virtual std::string toJson(void) const;

      virtual void resize_input(size_t _rawin_sz);

   protected:

      virtual const std::valarray<double>& calc_layer_output(const std::valarray<double>& _netin) = 0;

      /**
       * Calculate the net input value based on the raw input vector and weights specified in the
       * argument list and writes it into the _netin argument.
       */
      virtual const std::valarray<double>& calc_netin(const std::valarray<double>& _rawin) = 0;

      /*
       * Calculate and return the derivative of the basic_layer output with respect to
       * the net input for the most recent activation.
       */
      virtual const Array2D<double>& calc_dy_dnet(const std::valarray<double>& _out) = 0;

      /**
       * Calculate the derivative of the net input with respect to the weights based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      virtual const Array2D<double>& calc_dnet_dw(const std::valarray<double>& _rawin) = 0;

      /**
       * Calculate the derivative of the net input with respect to the raw input based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      virtual const Array2D<double>& calc_dnet_dx(const std::valarray<double>& _rawin) = 0;


      // Mark all derivative arrays as stale (e.g. after new activation).
      void stale(void);

      mutable std::mt19937_64 rand_engine;

   private:
      void copy(const BasicLayer& _basic_layer);

   public:
      const std::valarray<double>& const_value = layer_state.outputv;
      const size_t& const_layer_output_size_ref = layer_output_size;

   private:

      /* ********************************************************************
       * Private basic_layer configuration and provisioning data members
       */
      size_t layer_output_size;

   protected:
      /* ********************************************************************
       * Protected basic_layer state data members
       */
      size_t layer_input_size;

      LayerState layer_state;

   public:
      LayerWeights layer_weights;

   protected:

      /* ********************************************************************
       * Protected derivative information
       */
      mutable LayerDerivatives layer_derivatives;

   };

   inline size_t BasicLayer::size() const
   {
      return layer_output_size;
   }

   inline size_t BasicLayer::input_size() const
   {
      return layer_input_size;
   }

   inline const std::valarray<double>& BasicLayer::operator()() const
   {
      return layer_state.outputv;
   }

   inline const std::valarray<double>& BasicLayer::input_error(void) const
   {
      return layer_state.input_errorv;
   }

   inline const LayerState& BasicLayer::state(void) const
   {
      return layer_state;
   }

   inline const Array2D<double>& BasicLayer::get_dy_dnet(void) const
   {
      return layer_derivatives.dy_dnet;
   }

   inline const Array2D<double>& BasicLayer::get_dnet_dw(void) const
   {
      return layer_derivatives.dnet_dw;
   }

   inline const Array2D<double>& BasicLayer::get_dnet_dx(void) const
   {
      return layer_derivatives.dnet_dx;
   }

   inline const Array2D<double>& BasicLayer::get_dE_dw(void) const
   {
      return layer_derivatives.dE_dw;
   }

   inline void BasicLayer::resize_input(size_t _rawin_sz)
   {
      layer_input_size = _rawin_sz;

      layer_state.rawinv.resize(layer_input_size);
      layer_state.input_errorv.resize(layer_input_size);

      layer_weights.resize(layer_output_size, layer_input_size);
      layer_derivatives.resize(layer_output_size, layer_input_size);
   }

   inline std::string BasicLayer::toJson(void) const
   {
      return "";
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_BASICLAYER_H_ */