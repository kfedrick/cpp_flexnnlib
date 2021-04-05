//
// Created by kfedrick on 2/20/21.
//

#include "BaseNeuralNet.h"
#include <iostream>
#include <NRandArray2DInitializer.h>
#include <Globals.h>

using std::vector;
using flexnnet::BaseNeuralNet;
using flexnnet::BasicLayer;
using flexnnet::LayerWeights;
using flexnnet::NetworkWeights;
using flexnnet::ValarrMap;

BaseNeuralNet::BaseNeuralNet(const flexnnet::NetworkTopology& _topology)
   : network_topology(_topology),
     virtual_network_output_layer_ref(network_topology.get_network_output_layer())
{
   initialize_weights();
}

BaseNeuralNet::BaseNeuralNet(const BaseNeuralNet& _nnet)
   : network_topology(_nnet.network_topology),
     virtual_network_output_layer_ref(network_topology.get_network_output_layer())
{
   copy(_nnet);
}

BaseNeuralNet::~BaseNeuralNet()
{
}

void
BaseNeuralNet::copy(const BaseNeuralNet& _nnet)
{

}



void
BaseNeuralNet::reset(void)
{
   // TODO - implement
}

const ValarrMap&
BaseNeuralNet::activate(const ValarrMap& _externin)
{
   /*
    * Activate all network layers
    */
   vector<std::shared_ptr<NetworkLayer>>
      & ordered_layers = network_topology.get_ordered_layers();

   for (int i = 0; i < ordered_layers.size(); i++)
      const std::valarray<double>& temp = ordered_layers[i]->activate(_externin);

   /*
    * Marshal output layer values into a single vector using the virtual
    * virtual_network_output_layer_ref object.
    */
   virtual_network_output_layer_ref.activate(_externin);

   return virtual_network_output_layer_ref.input_value_map();
}

const void
BaseNeuralNet::backprop(const ValarrMap& _egradient)
{
   /*
    * Backprop through all network layers in reverse activation order
    */
   vector<std::shared_ptr<NetworkLayer>>
      & ordered_layers = network_topology.get_ordered_layers();
   for (int i = ordered_layers.size() - 1; i >= 0; i--)
      ordered_layers[i]->backprop(_egradient);
}

const LayerWeights&
BaseNeuralNet::get_weights(const std::string _layerid) const
{
   // TODO - Validate layer id
   const std::map<std::string, std::shared_ptr<NetworkLayer>>
      & network_layers = network_topology.get_layers();
   return network_layers.at(_layerid)->weights();
}

void
BaseNeuralNet::initialize_weights(void)
{
   std::function<Array2D<double>(unsigned int, unsigned int)> f2 = random_2darray<double>;

   const vector<std::shared_ptr<NetworkLayer>>
      & network_layers = network_topology.get_ordered_layers();
   for (auto& layer_ptr : network_layers)
   {
      layer_ptr->set_weight_initializer(f2);
      layer_ptr->initialize_weights();
      layer_ptr->set_biases(0.0);
   }
}

void
BaseNeuralNet::set_weights(const std::string _layerid, const LayerWeights& _weights)
{
   // TODO - Validate layer id
   const std::map<std::string, std::shared_ptr<NetworkLayer>>
      & network_layers = network_topology.get_layers();
   network_layers.at(_layerid)->weights().copy(_weights);
}

void
BaseNeuralNet::adjust_weights(const std::string _layerid, const Array2D<double>& _deltaw)
{
   // TODO - Validate layer id
   std::map<std::string, std::shared_ptr<NetworkLayer>>
      & network_layers = network_topology.get_layers();
   network_layers.at(_layerid)->weights().adjust_weights(_deltaw);
}

NetworkWeights
BaseNeuralNet::get_weights(void) const
{
   NetworkWeights network_weights;

   const vector<std::shared_ptr<NetworkLayer>>
      & network_layers = network_topology.get_ordered_layers();
   for (auto& layer_ptr : network_layers)
      network_weights[layer_ptr->name()].copy(layer_ptr->weights());

   return network_weights;
}



void
BaseNeuralNet::set_weights(const NetworkWeights& _weights)
{
   vector<std::shared_ptr<NetworkLayer>>
      & network_layers = network_topology.get_ordered_layers();
   for (auto& layer_ptr : network_layers)
      layer_ptr->weights().copy(_weights.at(layer_ptr->name()));
}

std::string
BaseNeuralNet::toJSON(void) const
{
   return "";
}
