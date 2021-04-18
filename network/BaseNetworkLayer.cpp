//
// Created by kfedrick on 4/10/21.
//

#include <sstream>

#include <BaseNetworkLayer.h>
#include <NetworkLayer.h>

using flexnnet::BaseNetworkLayer;

BaseNetworkLayer::BaseNetworkLayer()
{
}

BaseNetworkLayer::BaseNetworkLayer(const BaseNetworkLayer& _layer)
{
}

BaseNetworkLayer::~BaseNetworkLayer()
{
   // -- No body
}

void BaseNetworkLayer::clear()
{
   external_input_fields.clear();
   activation_connections.clear();
   backprop_connections.clear();
}

void
BaseNetworkLayer::add_external_input_field(const std::string& _field, size_t _sz)
{
   /*
    * If the field is already in the list then throw an exception; otherwise
    * add it to the list of external inputs used by this network basic_layer.
    */
   bool found = false;
   for (auto it = external_input_fields.begin(); it != external_input_fields.end(); it++)
   {
      if (it->field() == _field)
      {
         found = true;
         break;
      }
   }

   if (found)
   {
      static std::stringstream sout;
      sout << "Error : NeuralNetBuilder::add_external_input_field - connection from \""
           << _field << "\" to \"" << name() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }

   // If we got here, add the new connection to the list
   external_input_fields.push_back(ExternalInputRecord(_field, _sz, external_input_fields.size()));
   set_input_size(calc_input_size());
}

void BaseNetworkLayer::add_connection(const std::string& _cid, const std::shared_ptr<NetworkLayer>& _from, const LayerConnRecord::ConnectionType _type)
{
   // Determine which collection list the new connection will be added to.
   std::vector<LayerConnRecord>* conns;
   if (_cid == "activation")
      conns = &activation_connections;
   else if (_cid == "backprop")
      conns = &backprop_connections;
   else
   {
      static std::stringstream sout;
      sout << "Error : NeuralNetBuilder::add_connection() - "
           << "Unrecognized connections list id (" << _cid << ").\n";
      throw std::invalid_argument(sout.str());
   }

   /*
    * If a connection from the _from layer to the _to layer already
    * exist then throw an exception; otherwise add the new connection.
    */
   bool found = false;
   for (auto it = conns->begin(); it != conns->end(); it++)
   {
      std::cout << it->layer().name() << "\n";
      if (it->layer().name() == _from->name())
      {
         found = true;
         break;
      }
   }

   if (found)
   {
      static std::stringstream sout;
      sout << "Error : NeuralNetBuilder::add_" << _cid << "_connection() - from \""
           << _from->name() << "\" to \"" << name() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }

   // If we got here, add the new connection to the list
   conns->push_back(LayerConnRecord(_from, _type));

   if (_cid == "activation")
   {
      input_error_map[_from->name()] = std::valarray<double>(_from->size());
      set_input_size(calc_input_size());
   }
}

/**
 * Determine the current size of the virtual input vector for this
 * layer by adding up the sizes of all external input fields and
 * network layers providing input to this layer.
 *
 * @return - the sum of the sizes of the input vectors to this layer
 */
unsigned int BaseNetworkLayer::calc_input_size(void)
{
   unsigned int input_size = 0;

   for (auto& it : external_input_fields)
      input_size += it.size();

   for (auto& it : activation_connections)
      input_size += it.layer().size();

   return input_size;
}