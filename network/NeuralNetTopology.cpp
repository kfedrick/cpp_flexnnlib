//
// Created by kfedrick on 4/10/21.
//

#include "NeuralNetTopology.h"

using flexnnet::NeuralNetTopology;

NeuralNetTopology::NeuralNetTopology(void)
{
}

NeuralNetTopology::NeuralNetTopology(const NeuralNetTopology& _topo)
{
   clear();
   copy(_topo);
}

NeuralNetTopology::~NeuralNetTopology()
{}

void NeuralNetTopology::clear(void)
{
   network_layers.clear();
   ordered_layers.clear();
   network_output_layers.clear();
}

void NeuralNetTopology::copy(const NeuralNetTopology& _topo)
{
   // Shallow clone of all network layer layer objects
  clone_layers(_topo.network_layers);

   // Finish deep copy of network layers by reconstructing the layer
   // connection info
   for (auto it = network_layers.begin(); it != network_layers.end(); it++)
   {
      std::string id = it->first;
      std::shared_ptr<NetworkLayer>& network_layer = it->second;
      const std::shared_ptr<NetworkLayer>& src_network_layer = _topo.network_layers.at(id);

      // Copy layer activation/backprop connections
      copy_layer_connections(network_layer->activation_connections, src_network_layer->activation_connections);
      copy_layer_connections(network_layer->backprop_connections, src_network_layer->backprop_connections);

      // Copy external input fields
      network_layer->external_input_fields = src_network_layer->external_input_fields;

      network_layer->input_error_map = src_network_layer->input_error_map;

   }

   // Add the network layer copies to network_output_layers
   const std::vector<std::shared_ptr<NetworkLayer>>& netout_layers = _topo.network_output_layers;
   for (auto it = netout_layers.begin(); it != netout_layers.end(); it++)
   {
      std::string id = (*it)->name();
      network_output_layers.push_back(network_layers.at(id));
   }

   // Add the network layer copies to ordered_layers
   const std::vector<std::shared_ptr<NetworkLayer>>& olayers = _topo.ordered_layers;
   for (auto it = olayers.begin(); it != olayers.end(); it++)
   {
      std::string id = (*it)->name();
      ordered_layers.push_back(network_layers.at(id));
   }
}

void NeuralNetTopology::clone_layers(const NETWORK_LAYER_MAP_TYP& _layers)
{
   network_layers.clear();

   for (auto it = _layers.begin(); it != _layers.end(); it++)
   {
      std::string id = it->first;
      std::shared_ptr<NetworkLayer> src_layer_ptr = it->second;

      // Clone network layer
      network_layers[id] = src_layer_ptr->clone();
   }
}

void NeuralNetTopology::copy_layer_connections(std::vector<LayerConnRecord>& _to, const std::vector<LayerConnRecord>& _from)
{
   _to.clear();

   // Iterate through original layer connection records and create
   // a new entry in the local copy that references the local
   // instance of the network layer.
   //
   for (auto& it : _from)
   {
      std::string conn_id = it.layer().name();
      LayerConnRecord::ConnectionType ctype = it.get_connection_type();

      _to.push_back(LayerConnRecord(network_layers.at(conn_id), ctype));
   }
}