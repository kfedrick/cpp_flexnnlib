//
// Created by kfedrick on 5/12/19.
//

#include "BasicNeuralNet.h"

using namespace std;
using namespace flexnnet;

const Pattern &BasicNeuralNet::activate (const Pattern &ipattern)
{
   /*
    * Activate all network layers
    */
   for (int i = 0; i < network_layers.size (); i++)
   {
      // Get a network layer
      NetworkLayer *layer = network_layers[i];

      const std::vector<double> &invec = (*layer).coelesce_input (ipattern);
      layer->activate (invec);
   }

   network_output_pattern = network_output_layer.coelesce_input (ipattern);

   return network_output_pattern;
}