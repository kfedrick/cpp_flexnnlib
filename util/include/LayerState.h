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
      std::vector<double> outputv;
      std::vector<double> netinv;
      std::vector<double> rawinv;
      std::vector<double> backprop_errorv;
   };
}

#endif //FLEX_NEURALNET_LAYERSTATE_H_
