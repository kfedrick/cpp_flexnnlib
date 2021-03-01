//
// Created by kfedrick on 2/20/21.
//

#include "NetworkTopology.h"

using std::string;
using std::vector;
using std::map;

using flexnnet::BasicLayer;
using flexnnet::NetworkLayerImpl;
using flexnnet::NetworkTopology;
using flexnnet::LayerConnRecord;

NetworkTopology::NetworkTopology(const NNetIO_Typ& _xinput_sample)
{
   sample_extern_input = _xinput_sample;
}

NetworkTopology::NetworkTopology(const NetworkTopology& _topo)
{
   network_layers.clear();
   ordered_layers.clear();
   network_output_layers.clear();

   copy(_topo);
}

NetworkTopology::~NetworkTopology()
{}

NetworkTopology& NetworkTopology::operator=(const NetworkTopology& _topo)
{
   copy(_topo);
   return *this;
}

void NetworkTopology::copy(const NetworkTopology& _topo)
{
   // Clone all base layer objects
   std::map<std::string, std::shared_ptr<BasicLayer>> cloned_layers = clone_baselayers(_topo);

   // Make certain current network layer map is cleared then create
   // copies of the network layers and add them to the network layer map
   //
   for (auto it = _topo.network_layers.begin(); it != _topo.network_layers.end(); it++)
   {
      std::string id = it->first;
      const std::shared_ptr<NetworkLayerImpl>& network_layer = it->second;
      network_layers[id] = copy_netlayer(network_layer, cloned_layers);
   }

   // Add the network layer copies to network_output_layers
   const std::vector<std::shared_ptr<NetworkLayerImpl>>& netout_layers = _topo.get_output_layers();
   for (auto it = netout_layers.begin(); it != netout_layers.end(); it++)
   {
      std::string id = (*it)->name();
      network_output_layers.push_back(network_layers.at(id));
   }

   // Add the network layer copies to ordered_layers
   const std::vector<std::shared_ptr<NetworkLayerImpl>>& olayers = _topo.get_ordered_layers();
   for (auto it = olayers.begin(); it != olayers.end(); it++)
   {
      std::string id = (*it)->name();
      ordered_layers.push_back(network_layers.at(id));
   }

   // Copy sample external inputs
   sample_extern_input = _topo.sample_extern_input;

   update_network_output();
}


std::map<std::string, std::shared_ptr<BasicLayer>> NetworkTopology::clone_baselayers(const NetworkTopology& _topo)
{
   std::map<std::string, std::shared_ptr<BasicLayer>> cloned_layers;
   for (auto it = _topo.network_layers.begin(); it != _topo.network_layers.end(); it++)
   {
      std::string id = it->first;
      std::shared_ptr<NetworkLayerImpl> netlayer_ptr = it->second;

      // Clone basic layer and add new network layer
      cloned_layers[id] = netlayer_ptr->layer()->clone();
   }
   return cloned_layers;
}

std::shared_ptr<NetworkLayerImpl> NetworkTopology::copy_netlayer(const std::shared_ptr<NetworkLayerImpl>& _netlayer, const std::map<std::string, std::shared_ptr<BasicLayer>>& _baselayers)
{
   const std::string& id = _netlayer->name();

   // Create a copy of the network layer for base layer [id]
   std::shared_ptr<NetworkLayerImpl> netlayer_copy = std::make_shared<NetworkLayerImpl>(NetworkLayerImpl(_baselayers.at(id)));

   // Copy all of the activation and backprop connections
   netlayer_copy->activation_connections = copy_connections(_netlayer->activation_connections, _baselayers);
   netlayer_copy->backprop_connections = copy_connections(_netlayer->backprop_connections, _baselayers);

   // Copy external input fields
   netlayer_copy->external_input_fields = _netlayer->external_input_fields;

   netlayer_copy->virtual_input_vector = _netlayer->virtual_input_vector;

   return netlayer_copy;
}

std::vector<LayerConnRecord> NetworkTopology::copy_connections(const std::vector<LayerConnRecord>& _conns, const std::map<std::string, std::shared_ptr<BasicLayer>>& _baselayers)
{
   std::vector<LayerConnRecord> conns_copy;

   // Iterate through original layer connection records and create
   // a new entry in the copy that references the new copy of the base layer
   //
   for (auto i = 0; i<_conns.size(); i++)
   {
      std::string id = _conns[i].layer().name();
      LayerConnRecord::ConnectionType ctype = _conns[i].get_connection_type();
      conns_copy.push_back(LayerConnRecord(_baselayers.at(id), ctype));
   }
   return conns_copy;
}

/**
 * Add a connection to the basic_layer, _to, from the basic_layer, _from.
 *
 * @param _to - the name of the basic_layer to recieve input
 * @param _from - the name of the basic_layer to send its output
 */
void
NetworkTopology::add_layer_connection(const string& _to, const string& _from, LayerConnRecord::ConnectionType _type)
{
   /*
    * Check that _to and _from layers exist in the list of network layers.
    * If not throw an exception.
    */
   if (network_layers.find(_to) == network_layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NetworkTopology::add_external_input_field() - "
           << "Target basic_layer : \"" << _to.c_str() << "\" does not exist.\n";
      throw std::invalid_argument(sout.str());
   }

   if (network_layers.find(_from) == network_layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NetworkTopology::add_layer_connection() - "
           << "Layer does not exist, _from = \"" << _from.c_str() << "\".\n";
      throw std::invalid_argument(sout.str());
   }

   // Get from_dependencies so we can validate connection type can be made
   std::set<std::string> from_dependencies;
   std::set<std::string> to_dependencies;
   get_input_dependencies(from_dependencies, _from);
   get_input_dependencies(to_dependencies, _to);

   switch (_type)
   {
      case LayerConnRecord::Forward:add_forward_connection(*network_layers[_to], *network_layers[_from], _type, from_dependencies);
         break;

      case LayerConnRecord::Recurrent:add_recurrent_connection(*network_layers[_to], *network_layers[_from], _type, from_dependencies);
         break;

      case LayerConnRecord::Lateral:add_lateral_connection(*network_layers[_to], *network_layers[_from], _type, to_dependencies, from_dependencies);
         break;
   };

   // Always update the activation order after adding a new connection
   update_activation_order();
}

/**
 * Add a connection to the basic_layer, _to, from an external input vector.
 * @param _to
 * @param _vec
 */
void
NetworkTopology::add_external_input_field(const string& _to, const string& _field)
{
   /*
    * Check that _to basic_layer and external input field, _field, exist in the
    * list of network layers and in the external input sample, respectively.
    * If not throw an exception.
    */
   if (network_layers.find(_to) == network_layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NetworkTopology::add_external_input_field() - "
           << "Target basic_layer : \"" << _to.c_str() << "\" does not exist.\n";
      throw std::invalid_argument(sout.str());
   }

   if (sample_extern_input.find(_field) == sample_extern_input.end())
   {
      static std::stringstream sout;
      sout << "Error : NetworkTopology::add_external_input_field() - "
           << "External input field : \"" << _field.c_str()
           << "\" does not exist.\n";
      throw std::invalid_argument(sout.str());
   }

   network_layers[_to]->add_external_input_field(_field, sample_extern_input.at(_field).size());

   // Always update the activation order after adding a new connection
   update_activation_order();
}

void
NetworkTopology::add_forward_connection(NetworkLayerImpl& _to, NetworkLayerImpl& _from, LayerConnRecord::ConnectionType _type, std::set<
   std::string>& _from_dependencies)
{
   /*
    * In order to add a valid forward connection from basic_layer _from to basic_layer _to, the
    * _from basic_layer must not have a forward activation dependency on output from the
    * _to basic_layer as this would indicate a cycle. Likewise the _to and _from layers
    * must not be the same as this would cause a cycle.
    */
   if (_to.name() == _from.name() || _from_dependencies.find(_to.name()) != _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - Can't add Forward connection from  \""
           << _from.name().c_str() << "\" => \"" << _to.name().c_str() << "\" - "
           << " would create cycle." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   // If we got here it's OK to add forward connection
   _to.add_activation_connection(_from.layer(), _type);
   _from.add_backprop_connection(_to.layer(), _type);
}

void
NetworkTopology::add_recurrent_connection(NetworkLayerImpl& _to, NetworkLayerImpl& _from, LayerConnRecord::ConnectionType _type, std::set<
   std::string>& _from_dependencies)
{
   /*
    * To add a valid recurrent connection from basic_layer _from to basic_layer _to, the
    * _from basic_layer must already have a forward activation dependency from the
    * _to basic_layer, or the _to and _from basic_layer must be the same.
    */
   if (_to.name() != _from.name() && _from_dependencies.find(_to.name()) == _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - "
           << "Can't add Recurrent connection from  \""
           << _from.name().c_str() << "\" => \"" << _to.name().c_str() << "\" - "
           << " no forward depenencies." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   // If we got here it's OK to add recurrent connection
   _to.add_activation_connection(_from.layer(), _type);
   _from.add_backprop_connection(_to.layer(), _type);
}

void
NetworkTopology::add_lateral_connection(NetworkLayerImpl& _to, NetworkLayerImpl& _from, LayerConnRecord::ConnectionType _type, std::set<
   std::string>& _to_dependencies, std::set<std::string>& _from_dependencies)
{
   /*
    * To add a valid lateral connection from basic_layer _from to basic_layer _to, the
    * _to and _from basic_layer must be distinct and there must not be any existing
    * forward connection from either one to the other.
    */
   if (_to.name() == _from.name())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - "
           << "Can't add Lateral connection from  \""
           << _from.name().c_str() << "\" to itself." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   if (_to_dependencies.find(_from.name()) != _to_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - Can't add Lateral connection from  \""
           << _from.name().c_str() << "\" => \"" << _to.name().c_str() << "\" - "
           << " Forward connection already exist." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   if (_from_dependencies.find(_to.name()) != _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - Can't add Lateral connection from  \""
           << _to.name().c_str() << "\" => \"" << _from.name().c_str() << "\" - "
           << " Forward connection already exist." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   // If we got here it's OK to add lateral connections
   _to.add_activation_connection(_from.layer(), _type);
   _from.add_backprop_connection(_to.layer(), _type);
   _to.add_backprop_connection(_from.layer(), _type);
   _from.add_activation_connection(_to.layer(), _type);
}

/**
 * Return a set containing the names of layers connected to the basic_layer, _to, through
 * a chain of one or more forward connections.
 *
 * @param _dependencies
 * @param _name
 */
void NetworkTopology::get_input_dependencies(std::set<std::string>& _dependencies, const std::string& _from)
{
   const vector<LayerConnRecord>& activation_conn_list = network_layers[_from]->activation_connections;

// Add indirect dependencies by recursing on direct dependencies not already in dependency set_weights
   for (auto& record : activation_conn_list)
   {
      // Ignore recurrent connections
      if (record.is_recurrent())
         continue;

      /*
       * If this basic_layer is not already in our dependency list then
       * add it and recurse; otherwise we've visited this basic_layer already
       * so do nothing.
       */
      if (_dependencies.find(record.layer().name()) == _dependencies.end())
      {
         _dependencies.insert(record.layer().name());
         get_input_dependencies(_dependencies, record.layer().name());
      }
   }
}

void NetworkTopology::update_activation_order(void)
{
   ordered_layers.clear();
   for (auto& item : network_layers)
   {
      std::string layer_name = item.first;

      /*
       * Find the first (if any) layer already in the ordered activation list
       * that has a feedforward input dependency on the new layer and insert the
       * new layer just in front of it.
       */
      bool inserted = false;
      for (auto it = ordered_layers.begin(); it != ordered_layers.end(); ++it)
      {
         auto& ordered_layer_name = (*it)->name();

         // Get dependencies for ordered_layer_name
         std::set<std::string> dependencies;
         get_input_dependencies(dependencies, ordered_layer_name);

         // If new layer_name is in the list then break now and insert.
         if (dependencies.find(layer_name) != dependencies.end())
         {
            ordered_layers.insert(it, network_layers[layer_name]);
            inserted = true;
            break;
         }
      }

      // If it wasn't already inserted, add it to the end now
      if (!inserted)
         ordered_layers.push_back(network_layers[layer_name]);
   }
}

void NetworkTopology::update_network_output(void)
{
   network_output_layer.clear();

   for (auto it = network_output_layers.begin(); it != network_output_layers.end(); it++)
      network_output_layer.add_activation_connection((*it)->layer());
}