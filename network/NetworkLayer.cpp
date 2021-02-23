//
// Created by kfedrick on 2/21/21.
//

#include "NetworkLayer.h"

#include <iostream>
#include "flexnnet_networks.h"

using flexnnet::BasicLayer;
using flexnnet::NetworkLayer;
using flexnnet::LayerConnRecord;

NetworkLayer::NetworkLayer()
{
}

NetworkLayer::NetworkLayer(bool _is_output)
{
   output_layer_flag = _is_output;
}

NetworkLayer::NetworkLayer(std::shared_ptr<BasicLayer>& _layer, bool _is_output)
{
   basiclayer = _layer;
   output_layer_flag = _is_output;
}

NetworkLayer::NetworkLayer(std::shared_ptr<BasicLayer>&& _layer, bool _is_output)
{
   basiclayer = std::forward<std::shared_ptr<BasicLayer>>(_layer);
   output_layer_flag = _is_output;
}

NetworkLayer::~NetworkLayer() {}

void NetworkLayer::add_external_input_field(const std::string& _field)
{
   /*
    * If the field is already in the list then throw an exception; otherwise
    * add it to the list of external inputs used by this network basiclayer.
    */
   std::vector<std::string>::iterator it;
   it = std::find (external_input_fields.begin(), external_input_fields.end(), _field);
   if (it != external_input_fields.end())
   {
      static std::stringstream sout;
      sout << "Error : LayerInput::add_external_input_field() - connection from \""
           << _field.c_str() << "\" to \"" << basiclayer->name().c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      external_input_fields.push_back(_field);
   }
}

void
NetworkLayer::add_activation_connection(std::shared_ptr<BasicLayer>& _from, LayerConnRecord::ConnectionType _type)
{
   /*
    * If a connection already exist to this layer from the _from layer
    * throw an exception; otherwise add the new connection.
    */
   bool found = false;
   for (auto it = activation_connections.begin(); it != activation_connections.end(); it++)
   {
      if (it->layer().name() == _from->name())
      {
         found = true;
         break;
      }
   }

   if (found)
   {
      static std::stringstream sout;
      sout << "Error : NetworkLayer::insert_activation_connection() - forward activation connection from \""
           << _from->name().c_str() << "\" to \"" << basiclayer->name().c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      activation_connections.push_back(LayerConnRecord(_from, _type));
   }
}

void
NetworkLayer::add_backprop_connection(std::shared_ptr<BasicLayer>& _from, LayerConnRecord::ConnectionType _type)
{
   /*
    * If a connection already exist to this layer from the _from layer
    * throw an exception; otherwise add the new connection.
    */
   bool found = false;
   for (auto it = backprop_connections.begin(); it != backprop_connections.end(); it++)
   {
      if (it->layer().name() == _from->name())
      {
         found = true;
         break;
      }
   }

   if (found)
   {
      static std::stringstream sout;
      sout << "Error : NetworkLayer::insert_backprop_connection() - backprop connection from \""
           << _from->name().c_str() << "\" to \"" << basiclayer->name().c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      backprop_connections.push_back(LayerConnRecord(_from, _type));
   }
}

const std::valarray<double>& NetworkLayer::activate(const NNetIO_Typ& _externin)
{
   marshal_inputs(_externin);
   return basiclayer->activate(virtual_input_vector);
}

const std::valarray<double>& NetworkLayer::marshal_inputs(const NNetIO_Typ& _externin)
{
   size_t virtual_ndx = 0;

   // First append external input fields
   for (auto& inputrec : external_input_fields)
   {
      const std::valarray<double>& xinputv = _externin.at(inputrec);
      virtual_ndx = append_virtual_vector(virtual_ndx, xinputv);
   }

   // Append input layer outputs to the virtual input vector for this layer.
   for (size_t i = 0; i < activation_connections.size(); i++)
   {
      LayerConnRecord& conn = activation_connections[i];

      const BasicLayer& source_layer = conn.layer();
      const std::valarray<double>& layer_outputv = source_layer();

      virtual_ndx = append_virtual_vector(virtual_ndx, layer_outputv);
   }

   return virtual_input_vector;
}

size_t NetworkLayer::append_virtual_vector(size_t start_ndx, const std::valarray<double>& vec)
{
   size_t virtual_ndx = start_ndx;
   for (size_t ndx = 0; ndx < vec.size(); ndx++)
      virtual_input_vector[virtual_ndx++] = vec[ndx];
   return virtual_ndx;
}