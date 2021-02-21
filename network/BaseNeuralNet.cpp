//
// Created by kfedrick on 2/20/21.
//

#include "BaseNeuralNet.h"
#include <iostream>

using flexnnet::BaseNeuralNet;
using flexnnet::BasicLayer;

BaseNeuralNet::BaseNeuralNet(const std::string& _name, const std::vector<std::shared_ptr<BasicLayer>>& _layers)
   : NamedObject(_name), network_layers(_layers)
{
   std::map<std::string, std::valarray<double>> omap = init_network_output_layer();

   for (auto& item : _layers)
      layer_name_set.insert(item->name());
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

const flexnnet::NNetIO_Typ& BaseNeuralNet::activate(const NNetIO_Typ& _xdatum)
{
   /*
    * Activate all network layers
    */
   for (int i = 0; i < network_layers.size(); i++)
   {
      // Get a network layer
      BasicLayer& layer = *network_layers[i];

      //const std::valarray<double>& invec = layer.coelesce_input(_xdatum);
      //layer.activate(invec);
   }

   // Next add layer outputs
   for (auto nlayer : network_layers)
   {
/*
      if (nlayer->is_output_layer())
      {
         const std::valarray<double>& layer_outputv = (*nlayer)();
         network_output[nlayer->name()] = layer_outputv;
      }
*/
   }
   //return network_output_pattern;
   return network_output;
}

const void BaseNeuralNet::backprop(const std::valarray<double>& _gradient)
{

}

NetworkWeights BaseNeuralNet::get_weights(void) const
{
   NetworkWeights network_weights;

   for (auto alayer_ptr : network_layers)
      network_weights[alayer_ptr->name()].copy(alayer_ptr->layer_weights);

   return network_weights;
}

void BaseNeuralNet::set_weights(const NetworkWeights& _weights)
{
   for (auto alayer_ptr : network_layers)
      alayer_ptr->layer_weights.copy(_weights.at(alayer_ptr->name()));
}

std::string BaseNeuralNet::toJSON(void) const
{
   return "";
}
