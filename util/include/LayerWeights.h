//
// Created by kfedrick on 5/11/19.
//

#ifndef FLEX_NEURALNET_LAYERWEIGHTS_H_
#define FLEX_NEURALNET_LAYERWEIGHTS_H_

#include "NamedObject.h"
#include "Array2D.h"

namespace flexnnet
{
   /**
    * LayerWeights holds the learned parameters of the basiclayer, including:
    *
    *    weights - basiclayer inter-connection weights
    *    initial_layer_value - the initial value of the basiclayer neurons
    */
   class LayerWeights
   {
   public:

      /**
       * Null constructor
       */
      LayerWeights(void);

      /**
       * Initializing constructor
       */
      LayerWeights(const std::vector<std::vector<double>>& _lweights);

      /**
       * Copy constructor
       *
       * @param _lweights
       */
      LayerWeights(const LayerWeights& _lweights);

      /**
       * Copy constructor
       *
       * @param _lweights
       */
      LayerWeights(const LayerWeights&& _lweights);

      /**
       * Resize the basiclayer weights
       *
       * @param _layer_sz
       * @param _layer_input_sz
       */
      void resize(size_t _layer_sz, size_t _layer_input_sz);

      /**
       * Zero the basiclayer weights.
       */
      void zero(void);

      /**
       * Initialize basiclayer weights to specified value.
       */
      void set(const Array2D<double>& _weights);

      /**
       * Initialize basiclayer weights to specified value.
       *
       * @param _weights
       */
      void set(const std::vector<std::vector<double>>& _lweights);

      /**
       * Set the initial value of the neurons upon basiclayer reset.
       *
       * @param _ival
       */
      void set_initial_value(const std::valarray<double>& _ival);

      /**
       * Initialize basiclayer weights to specified value.
       */
      void copy(const LayerWeights& _weights);

      /**
       * Initialize basiclayer weights to specified value.
       */
      void copy(const LayerWeights&& _weights);

      /**
       * Assignment operator overload
       *
       * @param _weights
       * @return
       */
      LayerWeights& operator=(const LayerWeights& _lweights);

      /**
       * Assignment operator overload
       *
       * @param _weights
       * @return
       */
      LayerWeights& operator=(const std::vector<std::vector<double>>& _lweights);

      /**
       * Adjust basiclayer weights by the specified delta weight array.
       */
      void adjust_weights(const Array2D<double>& _delta);

      /**
       * Write biases and weights to json string
       */
      std::string to_json(void) const;

      /**
       * Read in biases and weights from json string
       * @param json
       */
      void from_json(const std::string& json);

   public:
      const Array2D<double>& const_weights_ref = weights;

   private:
      std::valarray<double> initial_layer_value;
      Array2D<double> weights;
   };

   inline void LayerWeights::zero(void)
   {
      weights = 0;
      initial_layer_value = 0;
   }

   inline void LayerWeights::set(const Array2D<double>& _weights)
   {
      weights.set(_weights);
   }

   inline void LayerWeights::set(const std::vector<std::vector<double>>& _lweights)
   {
      weights.set(_lweights);
   }

   inline void LayerWeights::set_initial_value(const std::valarray<double>& _ival)
   {
      initial_layer_value = _ival;
   }

   inline void LayerWeights::copy(const LayerWeights& _lweights)
   {
      weights.set(_lweights.weights);
      initial_layer_value = _lweights.initial_layer_value;
   }

   inline void LayerWeights::copy(const LayerWeights&& _lweights)
   {
      weights.set(std::forward<const Array2D<double>>(_lweights.weights));
      initial_layer_value = std::move(_lweights.initial_layer_value);
   }

   inline LayerWeights& LayerWeights::operator=(const LayerWeights& _lweights)
   {
      copy(_lweights);
      return *this;
   }

   inline LayerWeights& LayerWeights::operator=(const std::vector<std::vector<double>>& _lweights)
   {
      weights = _lweights;
   }

   inline void LayerWeights::adjust_weights(const Array2D<double>& _delta)
   {
      weights += _delta;
   }
}

#endif //FLEX_NEURALNET_LAYERWEIGHTS_H_
