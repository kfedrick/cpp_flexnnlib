//
// Created by kfedrick on 6/17/19.
//

#include "BasicNeuralNetSerializer.h"
#include "PureLin.h"
#include "LogSig.h"
#include "TanSig.h"
#include "SoftMax.h"
#include "RadBas.h"

#include "NetworkLayer.h"

#include <iostream>

using flexnnet::ExternalInputRecord;
using flexnnet::LayerConnRecord;
using flexnnet::BasicLayerSerializer;
using flexnnet::BasicNeuralNetSerializer;

using flexnnet::PureLin;
using flexnnet::LogSig;
using flexnnet::TanSig;
using flexnnet::SoftMax;
using flexnnet::RadBas;

using flexnnet::NetworkLayer;

std::shared_ptr<flexnnet::BasicNeuralNet> flexnnet::BasicNeuralNetSerializer::parse(const std::string& _json)
{
   rapidjson::Document netdoc;
   netdoc.Parse(_json.c_str());

   Datum network_input = parseNetworkInput(netdoc["network_input"].GetArray());

   std::map<std::string, std::shared_ptr<NetworkLayer>> layers;
   layers = parseNetworkLayers(netdoc["network_layers"].GetArray());

   std::vector<std::shared_ptr<NetworkLayer>> network_layers;
   network_layers = parseNetworkTopology(netdoc["layer_topology"].GetArray(), layers, network_input);

   std::shared_ptr<BasicNeuralNet> net = std::shared_ptr<BasicNeuralNet>(new BasicNeuralNet(network_layers, false));

   return net;
}

std::vector<std::shared_ptr<flexnnet::NetworkLayer>>
flexnnet::BasicNeuralNetSerializer::parseNetworkTopology(const rapidjson::Value& _obj, std::map<std::string,
                                                                                                std::shared_ptr<
                                                                                                   NetworkLayer>>& _layers, const Datum& _network_input)
{
   std::vector<std::shared_ptr<NetworkLayer>> network_layers;
   std::map<std::string, LayerInputInfo> topology;

   // Iterate through each layer connection entry
   //
   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      LayerInputInfo layerinfo;

      // These connections are for layer "id"
      std::string id = _obj[i]["id"].GetString();

      // Create network layer
      //NetworkLayer network_layer(_layers[id]);
      //NetworkLayer network_layer(_layers[id]->size(), _layers[id]->name(), BasicLayer::Output);
      std::shared_ptr<NetworkLayer> network_layer = _layers[id];

      // Get the layers external inputs and add them to the network layer
      layerinfo = parseExternalInput(_obj[i]["external_inputs"]);
      network_layer->add_external_input(_network_input, layerinfo.external_input_fields);

      // Add the input connections from other layers.
      layerinfo = parseLayerConnections(_obj[i]["input_connections"]);
      for (auto& item : layerinfo.layer_conn_info)
         network_layer->add_connection(*_layers[item.get_input_layer_id()], item.get_connection_type());

      network_layers.push_back(network_layer);
   }

   return network_layers;
}

flexnnet::Datum flexnnet::BasicNeuralNetSerializer::parseNetworkInput(const rapidjson::Value& _obj)
{
   std::map<std::string, std::valarray<double> > datum_fields;
   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      std::string field = _obj[i]["field"].GetString();
      size_t field_sz = _obj[i]["size"].GetUint64();
      size_t field_index = _obj[i]["index"].GetUint64();

      datum_fields[field] = std::valarray<double>(field_sz);
   }

   Datum network_input(datum_fields);
   return network_input;
}

flexnnet::BasicNeuralNetSerializer::LayerInputInfo
flexnnet::BasicNeuralNetSerializer::parseExternalInput(const rapidjson::Value& _obj)
{
   LayerInputInfo layer_input_info;

   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      std::string field = _obj[i]["field"].GetString();
      size_t field_sz = _obj[i]["size"].GetUint64();
      size_t field_index = _obj[i]["index"].GetUint64();

      layer_input_info.external_input_fields.insert(field);
      layer_input_info.external_inputs.push_back(ExternalInputRecord(field, field_sz, field_index));
   }

   return layer_input_info;
}
flexnnet::BasicNeuralNetSerializer::LayerInputInfo
flexnnet::BasicNeuralNetSerializer::parseLayerConnections(const rapidjson::Value& _obj)
{
   LayerInputInfo layer_input_info;
   std::vector<LayerConnRecord> input_layers;

   std::string layer_id;
   std::string conn_type_str;
   size_t layer_sz;

   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      LayerConnRecord ilayer_info;

      std::string id = _obj[i]["id"].GetString();
      size_t input_size = _obj[i]["size"].GetUint64();
      LayerConnRecord::ConnectionType connection_type = StringToConnType(_obj[i]["conn_type"].GetString());

      layer_input_info.layer_conn_info.push_back(LayerConnRecord(id, input_size, connection_type));
   }

   return layer_input_info;
}
std::map<std::string, std::shared_ptr<flexnnet::NetworkLayer>>
flexnnet::BasicNeuralNetSerializer::parseNetworkLayers(const rapidjson::Value& _obj)
{
   std::map<std::string, std::shared_ptr<NetworkLayer>> layers;

   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      std::string layertype = _obj[i]["transfer_function"]["type"].GetString();

      if (layertype == "flexnnet::PureLin")
      {
         auto layer = LayerSerializer<PureLin>::parse(_obj[i]);
         layers[layer->name()] = layer;
      }
      else if (layertype == "flexnnet::LogSig")
      {
         auto layer = LayerSerializer<LogSig>::parse(_obj[i]);
         layers[layer->name()] = layer;
      }
      else if (layertype == "flexnnet::TanSig")
      {
         auto layer = LayerSerializer<TanSig>::parse(_obj[i]);
         layers[layer->name()] = layer;
      }
      else if (layertype == "flexnnet::SoftMax")
      {
         auto layer = LayerSerializer<SoftMax>::parse(_obj[i]);
         layers[layer->name()] = layer;
      }
      else if (layertype == "flexnnet::RadBas")
      {
         auto layer = LayerSerializer<RadBas>::parse(_obj[i]);
         layers[layer->name()] = layer;
      }
   }

   return layers;
}
flexnnet::LayerConnRecord::ConnectionType
flexnnet::BasicNeuralNetSerializer::StringToConnType(const std::string& _typeStr)
{
   if (_typeStr == "forward")
      return LayerConnRecord::Forward;
   else if (_typeStr == "recurrent")
      return LayerConnRecord::Recurrent;
   else if (_typeStr == "lateral")
      return LayerConnRecord::Lateral;

   static std::stringstream sout;
   sout << "Error : BasicLayerSerializer::StringToConnType() - Bad connection type string \""
        << _typeStr.c_str() << "\"." << std::endl;
   throw std::invalid_argument(sout.str());
}
