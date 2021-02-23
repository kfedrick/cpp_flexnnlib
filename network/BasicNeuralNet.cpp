//
// Created by kfedrick on 5/12/19.
//

#include "BasicNeuralNet.h"
#include "BasicNeuralNetSerializer.h"

#include <iostream>

using flexnnet::BasicNeuralNet;
using flexnnet::BasicLayer;

BasicNeuralNet::BasicNeuralNet(const std::vector<std::shared_ptr<OldNetworkLayer>>& _layers, bool _recurrent, const std::string& _name)
   : NamedObject(_name), network_layers(_layers), recurrent_network_flag(_recurrent)
{
   std::map<std::string, std::valarray<double>> omap = init_network_output_layer();
   //network_output_pattern.set_weights(omap);

   for (auto& item : _layers)
      layer_name_set.insert(item->name());
}

BasicNeuralNet::~BasicNeuralNet()
{
}

void BasicNeuralNet::initialize_weights(void)
{
   std::cout << "BasicNeuralNet::initialize_weights() - entry\n";

   // TODO - implement
}

void BasicNeuralNet::reset(void)
{
   // TODO - implement
}

std::map<std::string, std::valarray<double>> BasicNeuralNet::init_network_output_layer(void)
{
   size_t sz = 0;
   std::map<std::string, std::valarray<double>> opatt_map;

   for (auto& network_layer : network_layers)
      if (network_layer->is_output_layer())
      {
         sz += network_layer->size();
         network_output_conn.add_connection(*network_layer, OldLayerConnRecord::Forward);
         opatt_map[network_layer->name()] = std::valarray<double>(network_layer->size());
      }

   return opatt_map;
}

const flexnnet::NNetIO_Typ& BasicNeuralNet::activate(const NNetIO_Typ& _xdatum)
{
   /*
    * Activate all network layers
    */
   for (int i = 0; i < network_layers.size(); i++)
   {
      // Get a network basiclayer
      OldNetworkLayer& layer = *network_layers[i];

      const std::valarray<double>& invec = layer.coelesce_input(_xdatum);
      layer.activate(invec);
   }

   // Next add basiclayer outputs
   for (auto nlayer : network_layers)
   {
      if (nlayer->is_output_layer())
      {
         const std::valarray<double>& layer_outputv = (*nlayer)();
         network_output[nlayer->name()] = layer_outputv;
      }
   }
   //return network_output_pattern;
   return network_output;
}

const void BasicNeuralNet::backprop(const std::valarray<double>& _gradient)
{

}

NetworkWeights BasicNeuralNet::get_weights(void) const
{
   NetworkWeights network_weights;

   for (auto alayer_ptr : network_layers)
      network_weights[alayer_ptr->name()].copy(alayer_ptr->layer_weights);

   return network_weights;
}

void BasicNeuralNet::set_weights(const NetworkWeights& _weights)
{
   for (auto alayer_ptr : network_layers)
      alayer_ptr->layer_weights.copy(_weights.at(alayer_ptr->name()));
}

std::string BasicNeuralNet::toJSON(void) const
{
   return BasicNeuralNetSerializer::toJson(*this);
}