//
// Created by kfedrick on 5/11/19.
//

#ifndef FLEX_NEURALNET_LAYERWEIGHTS_H_
#define FLEX_NEURALNET_LAYERWEIGHTS_H_


#include "Array.h"

namespace flexnnet
{
   class LayerWeights
   {
   public:

      void resize(unsigned int _layer_sz, unsigned int _layer_input_sz);

      /**
       * Initialize layer weights to specified value.
       */
      void set_weights (const flexnnet::Array<double> &_weights);

      /**
       * Adjust layer weights by the specified delta weight array.
       */
      void adjust_weights (const flexnnet::Array<double> &_delta);

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
      const flexnnet::Array<double>& const_weights_ref = weights;

   private:
      flexnnet::Array<double> weights;
   };
}

#endif //FLEX_NEURALNET_LAYERWEIGHTS_H_
