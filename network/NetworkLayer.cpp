//
// Created by kfedrick on 2/28/21.
//

#include "NetworkLayer.h"

using flexnnet::ValarrMap;
using flexnnet::NetworkLayer;

NetworkLayer::NetworkLayer() {}

NetworkLayer::NetworkLayer(bool _is_output)
{
   output_layer_flag = _is_output;
}

NetworkLayer::NetworkLayer(const std::shared_ptr<BasicLayer>& _layer, bool _is_output)
{
   basic_layer = _layer;
   output_layer_flag = _is_output;
}

NetworkLayer::NetworkLayer(std::shared_ptr<BasicLayer>&& _layer, bool _is_output)
{
   basic_layer = std::forward<std::shared_ptr<BasicLayer>>(_layer);
   output_layer_flag = _is_output;
}

NetworkLayer::~NetworkLayer() {}

const std::valarray<double>& NetworkLayer::concat_inputs(const ValarrMap& _externin)
{
   size_t virtual_ndx = 0;

   // First append external input fields
   for (auto& inputrec : external_input_fields)
   {
      const std::valarray<double>& xinputv = _externin.at(inputrec.field());
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

const ValarrMap& NetworkLayer::marshal_inputs(const ValarrMap& _externin)
{
   // First append external input fields
   for (auto& inputrec : external_input_fields)
      input_map[inputrec.field()] = _externin.at(inputrec.field());

   // Append input layer outputs to the virtual input vector for this layer.
   for (size_t i = 0; i < activation_connections.size(); i++)
   {
      LayerConnRecord& conn = activation_connections[i];
      const BasicLayer& source_layer = conn.layer();
      input_map[source_layer.name()] = source_layer();
   }

   return input_map;
}


size_t NetworkLayer::append_virtual_vector(size_t start_ndx, const std::valarray<double>& vec)
{
   size_t virtual_ndx = start_ndx;
   for (size_t ndx = 0; ndx < vec.size(); ndx++)
      virtual_input_vector[virtual_ndx++] = vec[ndx];
   return virtual_ndx;
}

void NetworkLayer::add_external_input_field(const std::string& _field, size_t _sz)
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
      sout << "Error : LayerInput::add_external_input_field() - connection from \""
           << _field.c_str() << "\" to \"" << basiclayer()->name().c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      external_input_fields.push_back(ExternalInputRecord(_field, _sz, external_input_fields.size()));

      input_map[_field] = {};
      virtual_input_vector.resize(virtual_input_vector.size() + _sz);
      basic_layer->resize_input(virtual_input_vector.size());
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
      sout << "Error : NetworkLayerImpl::insert_activation_connection() - forward activation connection from \""
           << _from->name().c_str() << "\" to \"" << basiclayer()->name().c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      activation_connections.push_back(LayerConnRecord(_from, _type));

      input_map[_from->name()] = {};
      virtual_input_vector.resize(virtual_input_vector.size() + _from->size());
      if (basiclayer().use_count() > 0)
         basic_layer->resize_input(virtual_input_vector.size());
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
      sout << "Error : NetworkLayerImpl::insert_backprop_connection() - backprop connection from \""
           << _from->name().c_str() << "\" to \"" << basiclayer()->name().c_str() << "\" already exists.\n";
      throw std::invalid_argument(sout.str());
   }
   else
   {
      backprop_connections.push_back(LayerConnRecord(_from, _type));
   }
}