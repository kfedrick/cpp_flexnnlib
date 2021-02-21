//
// Created by kfedrick on 8/25/19.
//

#include "LayerInput.h"

using flexnnet::LayerInput;

size_t LayerInput::add_connection(BasicLayer& _layer, LayerConnRecord::ConnectionType _type)
{
   // If we already have a connection from this layer then throw exception.
   const std::string& x = _layer.name();
   if (input_layer_names.find(_layer.name()) != input_layer_names.end())
   {
      static std::stringstream sout;
      sout << "Error : LayerInput::add_connection() - connection from \""
           << _layer.name().c_str() << "\"" << "\" already exists." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   input_layers.push_back(LayerConnRecord(&_layer, _type));
   input_layer_names.insert(_layer.name());

   virtual_input_vector.resize(virtual_input_vector.size() + _layer.size());

   backprop_error_vector.push_back(std::valarray<double>(_layer.size()));

   return virtual_input_vector.size();
}

size_t LayerInput::add_external_input(const Datum& _xdatum, const std::set<std::string>& _indexSet)
{
   static std::stringstream sout;

   // If we already have an external input specified then throw exception.
   if (!external_inputs.empty())
   {
      sout << "Error : LayerConnection::add_external_input_field() - external input already set_weights."
           << std::endl;
      throw std::invalid_argument(sout.str());
   }

   size_t virtual_external_size = 0;
   std::set<std::string> keyset = _xdatum.key_set();
   for (auto& field : _indexSet)
   {
      if (keyset.find(field) == keyset.end())
      {
         sout << "Error : LayerConnection::add_external_input_field() - field \""
              << field << "\" not found in Datum." << std::endl;
         throw std::invalid_argument(sout.str());
      }

      external_inputs.push_back(ExternalInputRecord(field, _xdatum[field].size(), _xdatum.index(field)));
      backprop_error_vector.push_back(std::valarray<double>(_xdatum[field].size()));

      virtual_external_size += _xdatum[field].size();
   }
   virtual_input_vector.resize(virtual_input_vector.size() + virtual_external_size);

   return virtual_input_vector.size();
}

const std::valarray<double>& LayerInput::coelesce_input(const Datum& _xdatum)
{
   int sz = virtual_input_vector.size();
   size_t virtual_ndx = 0;

   // First add external fields
   for (auto& inputrec : external_inputs)
   {
      const std::valarray<double>& inputv = _xdatum[inputrec.get_index()];
      virtual_ndx = append_virtual_vector(virtual_ndx, inputv);
   }

   // Next add layer outputs
   for (size_t map_ndx = 0; map_ndx < input_layers.size(); map_ndx++)
   {
      LayerConnRecord& conn = input_layers[map_ndx];

      const BasicLayer& in_layer = conn.get_input_layer();

      const std::valarray<double>& layer_outputv = in_layer();
      virtual_ndx = append_virtual_vector(virtual_ndx, layer_outputv);
   }

   return virtual_input_vector;
}

void LayerInput::backprop_scatter(const std::valarray<double> _errorv)
{
   unsigned int sz = _errorv.size();

   // TODO - range checking, clean up implementation
   int errv_ndx = 0;
   for (size_t map_ndx = 0; map_ndx < input_layers.size(); map_ndx++)
   {
      LayerConnRecord& conn = input_layers[map_ndx];
      size_t backprop_errorv_sz;

      BasicLayer& in_layer = conn.get_input_layer();
      backprop_errorv_sz = in_layer.size();

      for (size_t backprop_errorv_ndx = 0; backprop_errorv_ndx < backprop_errorv_sz; backprop_errorv_ndx++)
         backprop_error_vector.at(map_ndx)[backprop_errorv_ndx] = _errorv[errv_ndx++];

      // Tell layer to accumulate the new errorv
      in_layer.accumulate_error(backprop_error_vector[map_ndx]);
   }
}

size_t LayerInput::append_virtual_vector(size_t start_ndx, const std::valarray<double>& vec)
{
   size_t virtual_ndx = start_ndx;
   for (size_t ndx = 0; ndx < vec.size(); ndx++)
      virtual_input_vector[virtual_ndx++] = vec[ndx];
   return virtual_ndx;
}
