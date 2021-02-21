//
// Created by kfedrick on 2/20/21.
//

#include "NetworkTopology.h"

using std::string;
using std::vector;
using std::map;
using flexnnet::NetworkTopology;

NetworkTopology::NetworkTopology(const std::map<std::string, std::vector<double>> _xinput_sample)
{
   sample_extern_input = _xinput_sample;
}

NetworkTopology::~NetworkTopology() {}

/**
 * Add a connection to the layer, _to, from the layer, _from.
 *
 * @param _to - the name of the layer to recieve input
 * @param _from - the name of the layer to send its output
 */
void
NetworkTopology::add_layer_connection(const string& _to, const string& _from, NetworkTopology::ConnectionType _type)
{
   /*
    * Check that _to and _from layers exist in the list of network layers.
    * If not throw an exception.
    */
   if (layers.find(_to) == layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NetworkTopology::add_external_input_field() - "
           << "Target layer : \"" << _to.c_str() << "\" does not exist.\n";
      throw std::invalid_argument(sout.str());
   }

   if (layers.find(_from) == layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NetworkTopology::add_layer_connection() - "
           << "Layer does not exist, _from = \"" << _from.c_str() << "\".\n";
      throw std::invalid_argument(sout.str());
   }

   // Get from_dependencies so we can validate connection type can be made
   std::set<std::string> from_dependencies;
   std::set<std::string> to_dependencies;
   getInputDependencies(from_dependencies, _from);
   getInputDependencies(to_dependencies, _to);

   switch (_type)
   {
      case Forward:
         add_forward_connection(_to, _from, _type, from_dependencies);
         break;

      case Recurrent:
         add_recurrent_connection(_to, _from, _type, from_dependencies);
         break;

      case Lateral:
         add_lateral_connection(_to, _from, _type, to_dependencies, from_dependencies);
         break;
   };

   // Always update the activation order after adding a new connection
   update_activation_order();
}

/**
 * Add a connection to the layer, _to, from an external input vector.
 * @param _to
 * @param _vec
 */
void
NetworkTopology::add_external_input_field(const string& _to, const string& _field)
{
   /*
    * Check that _to layer and external input field, _field, exist in the
    * list of network layers and in the external input sample, respectively.
    * If not throw an exception.
    */
   if (layers.find(_to) == layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NetworkTopology::add_external_input_field() - "
           << "Target layer : \"" << _to.c_str() << "\" does not exist.\n";
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

   /*
    * Check that there is an entry for the _to in the external input fields
    * connection map. If not add one.
    */
   if (external_input_fields_conn_map.find(_to) != external_input_fields_conn_map.end())
      external_input_fields_conn_map.insert({});

   /*
    * If a connection already exist to the _to layer from the field, _field,
    * then throw an exception; otherwise add the new connection.
    */
   vector<string>& external_input_fields_list = external_input_fields_conn_map[_to];

   std::vector<string>::iterator it;
   it = std::find (external_input_fields_list.begin(), external_input_fields_list.end(), _field);
   if (it != external_input_fields_list.end())
   {
      static std::stringstream sout;
      sout << "Error : LayerInput::add_external_input_field() - connection from \""
           << _field.c_str() << "\" to \"" << _to.c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      external_input_fields_list.push_back(_field);
   }
}

void NetworkTopology::add_forward_connection(const string& _to, const string& _from, ConnectionType _type, std::set<std::string>& _from_dependencies)
{
   /*
    * In order to add a valid forward connection from layer _from to layer _to, the
    * _from layer must not have a forward activation dependency on output from the
    * _to layer as this would indicate a cycle. Likewise the _to and _from layers
    * must not be the same as this would cause a cycle.
    */
   if (_to == _from || _from_dependencies.find(_to) != _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - Can't add Forward connection from  \""
           << _from.c_str() << "\" => \"" << _to.c_str() << "\" - "
           << " would create cycle." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   // If we got here it's OK to add forward connection
   insert_activation_connection(_to, _from, _type);
   insert_backprop_connection(_from, _to, _type);
}

void NetworkTopology::add_recurrent_connection(const string& _to, const string& _from, ConnectionType _type, std::set<std::string>& _from_dependencies)
{
   /*
    * To add a valid recurrent connection from layer _from to layer _to, the
    * _from layer must already have a forward activation dependency from the
    * _to layer, or the _to and _from layer must be the same.
    */
   if (_to != _from && _from_dependencies.find(_to) == _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - "
           << "Can't add Recurrent connection from  \""
           << _from.c_str() << "\" => \"" << _to.c_str() << "\" - "
           << " no forward depenencies." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   // If we got here it's OK to add recurrent connection
   insert_activation_connection(_to, _from, _type);
   insert_backprop_connection(_from, _to, _type);
}

void NetworkTopology::add_lateral_connection(const string& _to, const string& _from, ConnectionType _type, std::set<std::string>& _to_dependencies, std::set<std::string>& _from_dependencies)
{
   /*
    * To add a valid lateral connection from layer _from to layer _to, the
    * _to and _from layer must be distinct and there must not be any existing
    * forward connection from either one to the other.
    */
   if (_to == _from)
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - "
           << "Can't add Lateral connection from  \""
           << _from.c_str() << "\" to itself." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   if (_to_dependencies.find(_from) != _to_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - Can't add Lateral connection from  \""
           << _from.c_str() << "\" => \"" << _to.c_str() << "\" - "
           << " Forward connection already exist." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   if (_from_dependencies.find(_to) != _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NetworkTopology::add_layer_connection() - Can't add Lateral connection from  \""
           << _to.c_str() << "\" => \"" << _from.c_str() << "\" - "
           << " Forward connection already exist." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   // If we got here it's OK to add lateral connections
   insert_activation_connection(_to, _from, _type);
   insert_backprop_connection(_from, _to, _type);
   insert_activation_connection(_from, _to, _type);
   insert_backprop_connection(_to, _from, _type);
}

void NetworkTopology::insert_activation_connection(const string& _to, const string& _from, ConnectionType _type)
{
   /*
    * Check that there is an entry for the _to connection in the activation
    * connection map. If not add one.
    */
   if (activation_conn_map.find(_to) == activation_conn_map.end())
      activation_conn_map[_to] = {};

   /*
    * If a connection already exist to the _to layer from the _from layer
    * throw an exception; otherwise add the new connection.
    */
   vector<LayerConnRecord>& activation_conn_list = activation_conn_map[_to];

   bool found = false;
   for (auto it = activation_conn_list.begin(); it != activation_conn_list.end(); it++)
   {
      if (it->layer->name() == _from)
      {
         found = true;
         break;
      }
   }

   if (found)
   {
      static std::stringstream sout;
      sout << "Error : LayerInput::add_connection() - forward activation connection from \""
           << _from.c_str() << "\" to \"" << _to.c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      activation_conn_list.push_back(LayerConnRecord(layers[_from], _type));
   }
}

void NetworkTopology::insert_backprop_connection(const string& _to, const string& _from, ConnectionType _type)
{
   /*
    * Check that there is an entry for the _to connection in the activation
    * connection map. If not add one.
    */
   if (backprop_conn_map.find(_to) == backprop_conn_map.end())
      backprop_conn_map[_to] = {};

   /*
    * If a connection already exist to the _to layer from the _from layer
    * throw an exception; otherwise add the new connection.
    */
   vector<LayerConnRecord>& backprop_conn_list = backprop_conn_map[_to];

   bool found = false;
   for (auto it = backprop_conn_list.begin(); it != backprop_conn_list.end(); it++)
   {
      if (it->layer->name() == _from)
      {
         found = true;
         break;
      }
   }

   if (found)
   {
      static std::stringstream sout;
      sout << "Error : LayerInput::insert_backprop_connection() - backprop connection from \""
           << _from.c_str() << "\" back to \"" << _to.c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      backprop_conn_list.push_back(LayerConnRecord(layers[_from], _type));
   }
}

/**
 * Return a set containing the names of layers connected to the layer, _to, through
 * a chain of one or more forward connections.
 *
 * @param _dependencies
 * @param _name
 */
void NetworkTopology::getInputDependencies(std::set<std::string>& _dependencies, const std::string& _from)
{
   const vector<LayerConnRecord>& activation_conn_list = activation_conn_map[_from];

// Add indirect dependencies by recursing on direct dependencies not already in dependency set_weights
   for (auto& record : activation_conn_list)
   {
      // Ignore recurrent connections
      if (record.is_recurrent())
         continue;

      /*
       * If this layer is not already in our dependency list then
       * add it and recurse; otherwise we've visited this layer already
       * so do nothing.
       */
      if (_dependencies.find(record.layer->name()) == _dependencies.end())
      {
         _dependencies.insert(record.layer->name());
         getInputDependencies(_dependencies, record.layer->name());
      }
   }
}

void NetworkTopology::update_activation_order(void)
{
   ordered_layers.clear();
   for (auto& item : layers)
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
         getInputDependencies(dependencies, ordered_layer_name);

         // If new layer_name is in the list then break now and insert.
         if (dependencies.find(layer_name) != dependencies.end())
         {
            ordered_layers.insert(it, layers[layer_name]);
            inserted = true;
            break;
         }
      }

      // If it wasn't already inserted, add it to the end now
      if (!inserted)
         ordered_layers.push_back(layers[layer_name]);
   }
}