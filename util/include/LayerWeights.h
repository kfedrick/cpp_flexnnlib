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
    * LayerWeights holds the learned parameters of the layer, including:
    *
    *    weights - layer inter-connection weights
    *    initial_value - the initial value of the layer neurons
    */
   class LayerWeights : public NamedObject
   {
   public:

      /**
       * Null constructor
       */
      LayerWeights(void);

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
       * Resize the layer weights
       *
       * @param _layer_sz
       * @param _layer_input_sz
       */
      void resize(size_t _layer_sz, size_t _layer_input_sz);

      /**
       * Zero the layer weights.
       */
      void zero(void);

      /**
       * Initialize layer weights to specified value.
       */
      void set_weights(const Array2D<double>& _weights);

      /**
       * Set the initial value of the neurons upon layer reset.
       *
       * @param _ival
       */
      void set_initial_value(const std::valarray<double>& _ival);

      /**
       * Initialize layer weights to specified value.
       */
      void copy(const LayerWeights& _weights);

      /**
       * Initialize layer weights to specified value.
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
       * Adjust layer weights by the specified delta weight array.
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
      std::valarray<double> initial_value;
      Array2D<double> weights;
   };

   inline void LayerWeights::zero(void)
   {
      weights = 0;
      initial_value = 0;
   }

   inline void LayerWeights::set_weights(const Array2D<double>& _weights)
   {
      weights = _weights;
   }

   inline void LayerWeights::set_initial_value(const std::valarray<double>& _ival)
   {
      initial_value = _ival;
   }

   inline void LayerWeights::copy(const LayerWeights& _lweights)
   {
      rename(_lweights.name());
      weights.set(_lweights.weights);
      initial_value = _lweights.initial_value;
   }

   inline void LayerWeights::copy(const LayerWeights&& _lweights)
   {
      rename(_lweights.name());
      weights.set(std::forward<const Array2D<double>>(_lweights.weights));
      initial_value = std::move(_lweights.initial_value);
   }


   inline LayerWeights& LayerWeights::operator=(const LayerWeights& _lweights)
   {
      copy(_lweights);
      return *this;
   }

   inline void LayerWeights::adjust_weights(const Array2D<double>& _delta)
   {
      this->weights += _delta;
   }
}

#endif //FLEX_NEURALNET_LAYERWEIGHTS_H_
