//
// Created by kfedrick on 5/12/19.
//

#include "BasicNeuralNet.h"
#include "BasicNeuralNetSerializer.h"

#include <iostream>

using flexnnet::BasicNeuralNet;
using flexnnet::BasicLayer;
using flexnnet::Datum;

BasicNeuralNet::BasicNeuralNet (const std::vector<std::shared_ptr<NetworkLayer>> &_layers, bool _recurrent, const std::string &_name)
   : NamedObject(_name), network_layers (_layers), recurrent_network_flag (_recurrent)
{
   std::map<std::string, std::valarray<double>> omap = init_network_output_layer ();
   //network_output_pattern.set(omap);

   for (auto &item : _layers)
      layer_name_set.insert (item->name ());
}

BasicNeuralNet::~BasicNeuralNet ()
{
}

void BasicNeuralNet::initialize_weights (void)
{
   std::cout << "BasicNeuralNet::initialize_weights() - entry\n";

   // TODO - implement
}

void BasicNeuralNet::reset (void)
{
   // TODO - implement
}

std::map<std::string, std::valarray<double>> BasicNeuralNet::init_network_output_layer (void)
{
   size_t sz = 0;
   std::map<std::string, std::valarray<double>> opatt_map;

   for (auto &network_layer : network_layers)
      if (network_layer->is_output_layer ())
      {
         sz += network_layer->size();
         network_output_conn.add_connection (*network_layer, LayerConnRecord::Forward);
         opatt_map[network_layer->name()] = std::valarray<double>(network_layer->size());
      }

   return opatt_map;
}


const Datum &BasicNeuralNet::activate (const Datum &_xdatum)
{
   /*
    * Activate all network layers
    */
   for (int i = 0; i < network_layers.size (); i++)
   {
      // Get a network layer
      NetworkLayer& layer = *network_layers[i];

      const std::valarray<double> &invec = layer.coelesce_input(_xdatum);
      layer.activate (invec);
   }

   // Next add layer outputs
   for (auto nlayer : network_layers)
   {
      if (nlayer->is_output_layer ())
      {
         const std::valarray<double> &layer_outputv = (*nlayer)();
         //network_output_pattern.set (nlayer->name (), layer_outputv);
      }
   }
   //return network_output_pattern;
   return Datum();
}


std::string BasicNeuralNet::toJSON(void) const
{
   return BasicNeuralNetSerializer::toJson (*this);
}