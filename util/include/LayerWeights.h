//
// Created by kfedrick on 5/11/19.
//

#ifndef FLEX_NEURALNET_LAYERWEIGHTS_H_
#define FLEX_NEURALNET_LAYERWEIGHTS_H_


#include "Array2D.h"

namespace flexnnet
{
   class LayerWeights
   {
   public:

      void resize(size_t _layer_sz, size_t _layer_input_sz);

      /**
       * Initialize layer weights to specified value.
       */
      void set_weights (const flexnnet::Array2D<double> &_weights);

      /**
       * Adjust layer weights by the specified delta weight array.
       */
      void adjust_weights (const flexnnet::Array2D<double> &_delta);

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
      const flexnnet::Array2D<double>& const_weights_ref = weights;

   private:
      flexnnet::Array2D<double> weights;
   };
}

#endif //FLEX_NEURALNET_LAYERWEIGHTS_H_
