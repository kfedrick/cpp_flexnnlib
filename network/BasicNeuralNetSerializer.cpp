//
// Created by kfedrick on 6/8/19.
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

std::string BasicNeuralNetSerializer::toJson(const BasicNeuralNet& _neural_net)
{
   std::string json;

   // Get Json serialized neural network object
   rapidjson::Value network_json_obj;
   encode(network_json_obj, _neural_net);

   // Copy object into the document
   rapidjson::Document document;
   document.SetObject();
   document.CopyFrom(network_json_obj, allocator);

   // Create a json writer
   rapidjson::StringBuffer buffer;
   rapidjson::Writer<rapidjson::StringBuffer> jsonWriter(buffer);

   // Write the json
   document.Accept(jsonWriter);
   std::string json_string = buffer.GetString();

   return json_string;
}

rapidjson::Value& BasicNeuralNetSerializer::encode(rapidjson::Value& _obj, const BasicNeuralNet& _neural_net)
{
   _obj.SetObject();

   // First encode network name
   std::string str = _neural_net.name();
   rapidjson::Value nnidobj;
   nnidobj.SetString(str.c_str(), str.size(), allocator);
   _obj.AddMember("neuralnet_id", nnidobj, allocator);

   // Second encode network layers
   rapidjson::Value _network_layers_obj;
   const std::vector<std::shared_ptr<NetworkLayer>>& network_layers = _neural_net.get_layers();
   encodeNetworkLayers(_network_layers_obj, network_layers);

   _obj.AddMember("network_layers", _network_layers_obj, allocator);

   // Next encode layer connection information
   rapidjson::Value _network_conn_obj;
   encodeLayerTopology(_network_conn_obj, network_layers);

   _obj.AddMember("layer_topology", _network_conn_obj, allocator);

   // Encode network input sample
   rapidjson::Value _network_input_obj;


//  Create BasicNeuralNet function that returns the input (field name, index, size)
//  encodeNetworkInput(_network_input_obj, _neural_net.get_network_input ());

   _obj.AddMember("network_input", _network_input_obj, allocator);

   return _obj;
}

rapidjson::Value&
BasicNeuralNetSerializer::encodeNetworkLayers(rapidjson::Value& _obj, const std::vector<std::shared_ptr<NetworkLayer>>& _network_layers)
{
   rapidjson::Document layerdoc;

   _obj.SetArray();

   for (auto& netlayer : _network_layers)
   {
      // Get and parse the layer json string
      std::string layer_json = netlayer->toJson();
      layerdoc.Parse(layer_json.c_str());

      // Push the layer Value object unto the encoded network layers array
      _obj.PushBack(layerdoc.GetObject(), allocator);
   }

   return _obj;
}

rapidjson::Value&
BasicNeuralNetSerializer::encodeLayerTopology(rapidjson::Value& _obj, const std::vector<std::shared_ptr<NetworkLayer>>& _network_layers)
{
   _obj.SetArray();

   for (auto& netlayer : _network_layers)
   {
      rapidjson::Value _layerconnobj;
      encodeNetworkConnections(_layerconnobj, *netlayer);

      _obj.PushBack(_layerconnobj, allocator);
   }

   return _obj;
}

rapidjson::Value& BasicNeuralNetSerializer::encodeNetworkConnections(rapidjson::Value& _obj, const NetworkLayer& _layer)
{
   _obj.SetObject();

   std::string str = _layer.name();
   rapidjson::Value thislayeridobj;
   thislayeridobj.SetString(str.c_str(), str.size(), allocator);
   _obj.AddMember("id", thislayeridobj, allocator);

   rapidjson::Value layerobj;

   // Add layer input connections
   layerobj.SetArray();

   rapidjson::Value conn_obj;
   for (auto& conn : _layer.get_input_connections())
   {
      encodeConnectionFromLayer(conn_obj, conn);
      layerobj.PushBack(conn_obj, allocator);
   }

   _obj.AddMember("input_connections", layerobj, allocator);

   rapidjson::Value xinputobj;
   xinputobj.SetObject();
   encodeExternalLayerInput(xinputobj, _layer.get_external_inputs());
   _obj.AddMember("external_inputs", xinputobj, allocator);

   return _obj;
}

rapidjson::Value&
flexnnet::BasicNeuralNetSerializer::encodeConnectionFromLayer(rapidjson::Value& _obj, const LayerConnRecord& _conn)
{
   _obj.SetObject();

   rapidjson::Value externobj, inlayerobj, intype_strobj, conntype_strobj;
   std::string str;

   str = _conn.get_input_layer().name();
   inlayerobj.SetString(str.c_str(), str.size(), allocator);
   _obj.AddMember("id", inlayerobj, allocator);

   std::string conn_type_str = connTypeToString(_conn.get_connection_type());
   conntype_strobj.SetString(conn_type_str.c_str(), conn_type_str.size(), allocator);
   _obj.AddMember("connection_type", conntype_strobj, allocator);

   _obj.AddMember("size", _conn.size(), allocator);

   return _obj;
}

rapidjson::Value&
flexnnet::BasicNeuralNetSerializer::encodeExternalLayerInput(rapidjson::Value& _obj, const std::vector<
   ExternalInputRecord>& _xinput)
{
   _obj.SetArray();

   rapidjson::Value strobj;
   for (auto& a_input : _xinput)
   {
      rapidjson::Value externobj;
      externobj.SetObject();

      strobj.SetString(a_input.get_field().c_str(), a_input.get_field().size(), allocator);
      externobj.AddMember("field", strobj, allocator);
      externobj.AddMember("index", a_input.get_index(), allocator);
      externobj.AddMember("size", a_input.get_size(), allocator);

      _obj.PushBack(externobj, allocator);
   }

   return _obj;
}

rapidjson::Value& BasicNeuralNetSerializer::encodeNetworkInput(rapidjson::Value& _obj, const Datum& _datum)
{
   _obj.SetArray();

   rapidjson::Value strobj;
   for (auto& field : _datum.key_set())
   {
      rapidjson::Value datumfieldobj;
      datumfieldobj.SetObject();

      strobj.SetString(field.c_str(), field.size(), allocator);
      datumfieldobj.AddMember("field", strobj, allocator);
      datumfieldobj.AddMember("index", _datum.index(field), allocator);
      datumfieldobj.AddMember("size", _datum[field].size(), allocator);

      _obj.PushBack(datumfieldobj, allocator);
   }

   return _obj;
}

std::string flexnnet::BasicNeuralNetSerializer::connTypeToString(LayerConnRecord::ConnectionType _type)
{
   switch (_type)
   {
      case LayerConnRecord::ConnectionType::Forward:return "forward";
      case LayerConnRecord::ConnectionType::Recurrent:return "recurrent";
      case LayerConnRecord::ConnectionType::Lateral:return "lateral";
   };
}

