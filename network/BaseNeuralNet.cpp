//
// Created by kfedrick on 2/20/21.
//

#include "BaseNeuralNet.h"
#include <iostream>

using std::vector;
using flexnnet::BaseNeuralNet;
using flexnnet::BasicLayer;
using flexnnet::NetworkWeights;
using flexnnet::ValarrMap;


BaseNeuralNet::BaseNeuralNet(const flexnnet::NetworkTopology& _topology) : network_topology(_topology)
{
}

BaseNeuralNet::~BaseNeuralNet()
{
}

void BaseNeuralNet::initialize_weights(void)
{
   // TODO - implement
}

void BaseNeuralNet::reset(void)
{
   // TODO - implement
}

const ValarrMap& BaseNeuralNet::activate(const ValarrMap& _externin)
{
   /*
    * Activate all network layers
    */
   vector<std::shared_ptr<NetworkLayerImpl>>& ordered_layers = network_topology.get_ordered_layers();
   for (int i = 0; i < ordered_layers.size(); i++)
      ordered_layers[i]->activate(_externin);

   /*
    * Marshal output layer values into a single vector using the virtual
    * network_output_layer object.
    */
   network_output_layer.activate(_externin);
   return network_output_layer.input_value_map();
}

const void BaseNeuralNet::backprop(const std::valarray<double>& _gradient)
{

}

NetworkWeights BaseNeuralNet::get_weights(void) const
{
   NetworkWeights network_weights;

   const vector<std::shared_ptr<NetworkLayerImpl>>& network_layers = network_topology.get_ordered_layers();
   for (auto& layer_ptr : network_layers)
      network_weights[layer_ptr->name()].copy(layer_ptr->weights());

   return network_weights;
}

void BaseNeuralNet::set_weights(const NetworkWeights& _weights)
{
   vector<std::shared_ptr<NetworkLayerImpl>>& network_layers = network_topology.get_ordered_layers();
   for (auto& layer_ptr : network_layers)
      layer_ptr->weights().copy(_weights.at(layer_ptr->name()));
}

std::string BaseNeuralNet::toJSON(void) const
{
   return "";
}
