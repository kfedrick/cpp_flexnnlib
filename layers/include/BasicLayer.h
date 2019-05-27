//
// Created by kfedrick on 5/8/19.
//

#ifndef FLEX_NEURALNET_BASICLAYER_H_
#define FLEX_NEURALNET_BASICLAYER_H_

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
      BasicLayer (unsigned int sz, const std::string &_name);

   public:
      virtual ~BasicLayer ();

      /* ********************************************************************
       * Initialization and provisioning methods
       */

   public:
      /* ********************************************************************
       * Public configuration and provisioning methods
       */

      /*
       * Specify the expected input vector size for this layer
       */
      virtual void resize_input_vector (unsigned int sz);

      /*
       * Return length of layer output vector
       */
      virtual unsigned int size () const;

      /*
       * Return length of layer input vector
       */
      virtual unsigned int input_size () const;

      /* ********************************************************************
       * Public layer state getter methods
       */
      const Array<double>& get_dAdN(void);
      const Array<double>& get_dNdW(void);
      const Array<double>& get_dNdI(void);

      /* ********************************************************************
       * Public layer operational methods
       */

      /**
       * Calculate the value of the layer neuron vector based on the specified
       * raw input vectors.
       * @param inputVec
       * @return
       */
      virtual const std::vector<double> &activate (const std::vector<double> &inputVec);

      /**
       * Accumulate error specified in _errorv into the current layer error
       * @param _errorv
       * @return
       */
      virtual const std::vector<double> &accumulate_error (const std::vector<double> &_errorv);

      /*
       * Return the current value of the network layer as a std::vector
       */
      virtual const std::vector<double> &operator() () const;

      std::function<std::string (void)> toJSONString;




   protected:

      std::function<const std::vector<double>& (const std::vector<double> &_rawin, const flexnnet::Array<double> &_weights)> calc_netin;

      std::function<void (std::vector<double>& _out, const std::vector<double>& _netin)> calc_layer_output;

      /**
       * Calculate and return the derivative of the layer net input with respect to
       * the layer weights for the most recent activation.
       */
      std::function<const Array<double>& (const std::vector<double> &_netin, const std::vector<double> &_rawin, const flexnnet::Array<
         double> &_weights)> calc_dNdW;

      /**
       * Calculate and return the derivative of the layer net input with respect to
       * the layer inputs for the most recent activation.
       */
      std::function<const Array<double>& (const std::vector<double> &_netin, const std::vector<double> &_rawin, const flexnnet::Array<
         double> &_weights)> calc_dNdI;

      /*
       * Calculate and return the derivative of the layer output with respect to
       * the net input for the most recent activation.
       */
      std::function<const Array<double>& (const std::vector<double>& _out, const std::vector<double>& _netin)> calc_dAdN;

      std::function<void (unsigned int _layer_sz, unsigned int _rawin_sz)> resize_layer;

      /**
       * Mark all derivative arrays as stale (e.g. after new activation).
       */
      void stale (void);


   public:
      const unsigned int& const_layer_output_size_ref = layer_output_size;
      const unsigned int& const_layer_input_size_ref = layer_input_size;


   protected:
      unsigned int layer_input_size;

   private:

      /* ********************************************************************
       * Private layer configuration and provisioning data members
       */
      const unsigned int layer_output_size;

   protected:
      /* ********************************************************************
       * Protected layer state data members
       */

      LayerState layer_state;

   public:
      LayerWeights layer_weights;

   protected:

      /* ********************************************************************
       * Protected derivative information
       */
      LayerDerivatives layer_derivatives;

   };

   /*
    * Return length of layer output vector
    */
   inline unsigned int BasicLayer::size () const
   {
      return layer_output_size;
   }

   /*
    * Return length of layer output vector
    */
   inline unsigned int BasicLayer::input_size () const
   {
      return layer_input_size;
   }

   inline const Array<double>& BasicLayer::get_dAdN(void)
   {
      if (layer_derivatives.stale_dAdN)
         layer_derivatives.dAdN = calc_dAdN(layer_state.outputv, layer_state.netinv);

      layer_derivatives.stale_dAdN = false;
      return layer_derivatives.dAdN;
   }

   inline const Array<double>& BasicLayer::get_dNdW(void)
   {
      if (layer_derivatives.stale_dNdW)
         layer_derivatives.dNdW = calc_dNdW(layer_state.netinv, layer_state.rawinv, layer_weights.const_weights_ref);

      layer_derivatives.stale_dNdW = false;
      return layer_derivatives.dNdW;
   }

   inline const Array<double>& BasicLayer::get_dNdI(void)
   {
      if (layer_derivatives.stale_dNdI)
         layer_derivatives.dNdI = calc_dNdI(layer_state.netinv, layer_state.rawinv, layer_weights.const_weights_ref);

      layer_derivatives.stale_dNdI = false;
      return layer_derivatives.dNdI;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_BASICLAYER_H_ */
