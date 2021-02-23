//
// Created by kfedrick on 2/20/21.
//

#include "BaseNeuralNet.h"
#include <iostream>

using std::vector;
using flexnnet::BaseNeuralNet;
using flexnnet::BasicLayer;

BaseNeuralNet::BaseNeuralNet(flexnnet::NetworkTopology& _topology)
{
   network_topology = std::shared_ptr<NetworkTopology>(&_topology);
}

BaseNeuralNet::~BaseNeuralNet()
{
}

void BaseNeuralNet::initialize_weights(void)
{
   std::cout << "BaseNeuralNet::initialize_weights() - entry\n";

   // TODO - implement
}

void BaseNeuralNet::reset(void)
{
   // TODO - implement
}

std::map<std::string, std::valarray<double>> BaseNeuralNet::init_network_output_layer(void)
{
   size_t sz = 0;
   std::map<std::string, std::valarray<double>> opatt_map;
/*

   for (auto& network_layer : network_layers)
      if (network_layer->is_output_layer())
      {
         sz += network_layer->size();
         network_output_conn.add_connection(*network_layer, LayerConnRecord::Forward);
         opatt_map[network_layer->name()] = std::valarray<double>(network_layer->size());
      }
*/

   return opatt_map;
}

const std::valarray<double>& BaseNeuralNet::activate(const NNetIO_Typ& _externin)
{
   /*
    * Activate all network layers
    */
   vector<std::shared_ptr<NetworkLayer>>& network_layers = network_topology->get_ordered_layers();
   for (int i = 0; i < network_layers.size(); i++)
      network_layers[i]->activate(_externin);

   /*
    * Marshal output layer values into a single vector using the virtual
    * network_output_layer object.
    */
   network_output_layer.activate(_externin);

   return network_output_layer.layer()->const_value;
}

const void BaseNeuralNet::backprop(const std::valarray<double>& _gradient)
{

}

NetworkWeights BaseNeuralNet::get_weights(void) const
{
   NetworkWeights network_weights;

   vector<std::shared_ptr<NetworkLayer>>& network_layers = network_topology->get_ordered_layers();
   for (auto& layer_ptr : network_layers)
      network_weights[layer_ptr->name()].copy(layer_ptr->layer()->layer_weights);

   return network_weights;
}

void BaseNeuralNet::set_weights(const NetworkWeights& _weights)
{
   vector<std::shared_ptr<NetworkLayer>>& network_layers = network_topology->get_ordered_layers();
   for (auto& layer_ptr : network_layers)
      layer_ptr->layer()->layer_weights.copy(_weights.at(layer_ptr->name()));
}

std::string BaseNeuralNet::toJSON(void) const
{
   return "";
}
