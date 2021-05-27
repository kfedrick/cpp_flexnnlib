//
// Created by kfedrick on 5/11/19.
//

#ifndef FLEX_NEURALNET_LAYERWEIGHTS_H_
#define FLEX_NEURALNET_LAYERWEIGHTS_H_

#include <functional>
#include <stdexcept>
#include <iostream>

#include "NamedObject.h"
#include "Array2D.h"

namespace flexnnet
{
   /**
    * LayerWeights holds the learned parameters of the basic_layer, including:
    *
    *    weights - basic_layer inter-connection weights
    *    initial_layer_value - the initial vectorize of the basic_layer neurons
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
       * Initializing constructor
       */
      LayerWeights(const Array2D<double>& _lweights);

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
       * Return dimensionality of the layer weight matrix
       * @return
       */
      Array2D<double>::Dimensions size(void) const;

      /**
       * Resize the basic_layer weights
       *
       * @param _layer_sz
       * @param _layer_input_sz
       */
      void resize(size_t _layer_sz, size_t _layer_input_sz);

      /**
       * Zero the basic_layer weights.
       */
      void zero(void);

      void set(double _val);

      /**
       * Initialize basic_layer weights to specified vectorize.
       */
      void set(const Array2D<double>& _weights);

      /**
       * Initialize basic_layer weights to specified vectorize.
       *
       * @param _weights
       */
      void set(const std::vector<std::vector<double>>& _lweights);

      /**
       * Set the bias values.
       *
       * @param _ival
       */
      void set_biases(double _val);

      /**
       * Set the bias values.
       *
       * @param _ival
       */
      void set_biases(const std::valarray<double>& _biases);

      /**
       * Set the initial vectorize of the neurons upon basic_layer clear.
       *
       * @param _ival
       */
      void set_initial_value(const std::valarray<double>& _ival);

      /**
       * Initialize basic_layer weights to specified vectorize.
       */
      void copy(const LayerWeights& _weights);

      /**
       * Initialize basic_layer weights to specified vectorize.
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
       * Adjust basic_layer weights by the specified delta weight array.
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

   inline Array2D<double>::Dimensions LayerWeights::size(void) const
   {
      return weights.size();
   }

   inline void LayerWeights::zero(void)
   {
      weights = 0;
      initial_layer_value = 0;
   }

   inline void LayerWeights::set(double _val)
   {
      weights.set(_val);
   }

   inline void LayerWeights::set(const Array2D<double>& _weights)
   {
      weights.assign(_weights);
   }

   inline void LayerWeights::set(const std::vector<std::vector<double>>& _lweights)
   {
      weights.set(_lweights);
   }

   inline
   void LayerWeights::set_biases(double _val)
   {
      Array2D<double>::Dimensions dims = weights.size();

      for (unsigned int i = 0; i<dims.rows; i++)
         weights.at(i,dims.cols-1) = _val;
   }

   inline
   void LayerWeights::set_biases(const std::valarray<double>& _biases)
   {
      Array2D<double>::Dimensions dims = weights.size();

      if (_biases.size() != dims.rows)
      {
         std::ostringstream err_str;
         err_str
            << "Error : LayerWeights::set_biases size - argument size : "
            << _biases.size() << " doesn't match expected (" << dims.cols << ").\n";
         throw std::invalid_argument(err_str.str());
      }

      for (unsigned int i = 0; i<dims.rows; i++)
         weights.at(i,dims.cols-1) = _biases[i];
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
