//
// Created by kfedrick on 5/13/19.
//

#ifndef FLEX_NEURALNET_LAYERSTATE_H_
#define FLEX_NEURALNET_LAYERSTATE_H_

#include "BasicLayer.h"

namespace flexnnet
{
   class LayerState
   {
   public:
      std::valarray<double> outputv;
      std::valarray<double> netinv;
      std::valarray<double> rawinv;
      std::valarray<double> backprop_errorv;
   };
}

#endif //FLEX_NEURALNET_LAYERSTATE_H_
