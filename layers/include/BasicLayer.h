//
// Created by kfedrick on 5/8/19.
//

#ifndef FLEX_NEURALNET_BASICLAYER_H_
#define FLEX_NEURALNET_BASICLAYER_H_

#include <set>
#include <unordered_set>
#include <sstream>
#include <functional>

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

   public:
      virtual ~BasicLayer();

   public:

      // Return length of layer output valarray
      size_t size() const;
      virtual size_t input_size() const;

      const Array2D<double>& get_dAdN(void) const;
      const Array2D<double>& get_dNdW(void) const;
      const Array2D<double>& get_dNdI(void) const;

   public:
      /* ********************************************************************
       * Public layer operational methods
       */

      /**
       * Calculate the value of the layer neuron vector based on the specified
       * raw input vectors.
       * @param inputVec
       * @return
       */
      virtual const std::valarray<double>& activate(const std::valarray<double>& inputVec);

      /**
       * Accumulate error specified in _errorv into the current layer error
       * @param _errorv
       * @return
       */
      virtual const std::valarray<double>& accumulate_error(const std::valarray<double>& _errorv);

      /*
       * Return the current value of the network layer as a std::valarray
       */
      virtual const std::valarray<double>& operator()() const;

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
       * Calculate and return the derivative of the layer output with respect to
       * the net input for the most recent activation.
       */
      virtual const Array2D<double>& calc_dAdN(const std::valarray<double>& _out) = 0;

      /**
       * Calculate the derivative of the net input with respect to the weights based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      virtual const Array2D<double>& calc_dNdW(const std::valarray<double>& _rawin) = 0;

      /**
       * Calculate the derivative of the net input with respect to the raw input based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      virtual const Array2D<double>& calc_dNdI(const std::valarray<double>& _rawin) = 0;

      // Mark all derivative arrays as stale (e.g. after new activation).
      void stale(void);

   public:
      const size_t& const_layer_output_size_ref = layer_output_size;

   private:

      /* ********************************************************************
       * Private layer configuration and provisioning data members
       */
      const size_t layer_output_size;

   protected:
      /* ********************************************************************
       * Protected layer state data members
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


   inline const Array2D<double>& BasicLayer::get_dAdN(void) const
   {
      return layer_derivatives.dAdN;
   }

   inline const Array2D<double>& BasicLayer::get_dNdW(void) const
   {
      return layer_derivatives.dNdW;
   }

   inline const Array2D<double>& BasicLayer::get_dNdI(void) const
   {
      return layer_derivatives.dNdI;
   }

   inline void BasicLayer::resize_input(size_t _rawin_sz)
   {
      layer_input_size = _rawin_sz;
      layer_weights.resize(layer_output_size, layer_input_size);
      layer_derivatives.resize(layer_output_size, layer_input_size);
   }

   inline std::string BasicLayer::toJson(void) const
   {
      return "";
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_BASICLAYER_H_ */