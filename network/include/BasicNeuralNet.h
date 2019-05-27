//
// Created by kfedrick on 5/12/19.
//

#ifndef FLEX_NEURALNET_BASICNEURALNET_H_
#define FLEX_NEURALNET_BASICNEURALNET_H_

#include <map>
#include "NetworkLayer.h"
#include "Pattern.h"

namespace flexnnet
{
   class BasicNeuralNet
   {
   public:
      BasicNeuralNet (const std::string &_name  = "BasicNeuralNet");
      virtual ~BasicNeuralNet ();

      virtual const Pattern &activate(const Pattern &ipattern);

   private:

      // Network layers stored in proper activation order
      std::vector<NetworkLayer*> network_layers;

      NetworkLayer network_output_layer;

      Pattern network_output_pattern;

      Pattern pattern_placeholder;
   };
}

#endif //FLEX_NEURALNET_BASICNEURALNET_H_
