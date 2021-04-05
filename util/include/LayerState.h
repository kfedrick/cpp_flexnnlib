//
// Created by kfedrick on 5/13/19.
//

#ifndef FLEX_NEURALNET_LAYERSTATE_H_
#define FLEX_NEURALNET_LAYERSTATE_H_

namespace flexnnet
{
   class LayerState
   {
   public:
      // The cached value of the most recent raw input value.
      std::valarray<double> rawinv;

      // The net input value (e.g. net sum of the layer input)
      std::valarray<double> netinv;

      // The layer output value.
      std::valarray<double> outputv;


      // The cached value of the most recent external layer error.
      std::valarray<double> external_errorv;

      // Partial derivative of the external layer error with respect
      // to the net input value (e.g. net sum).
      std::valarray<double> netin_errorv;

      // Partial derivative of the external layer error with respect
      // to the raw input.
      std::valarray<double> input_errorv;
   };
}

#endif //FLEX_NEURALNET_LAYERSTATE_H_
