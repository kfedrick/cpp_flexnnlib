//
// Created by kfedrick on 5/26/19.
//
#include <typeinfo>
#include <cxxabi.h>
#include <iostream>
#include "NetworkLayerSerializer.h"

using flexnnet::NetworkLayerSerializer;

flexnnet::NetworkLayerSerializer::NetworkLayerSerializer()
{

}

flexnnet::NetworkLayerSerializer::~NetworkLayerSerializer()
{

}


std::string flexnnet::NetworkLayerSerializer::demangle(const std::string& name)
{
   static char buf[1024];
   size_t size = sizeof(buf);
   int status;

   char* res = abi::__cxa_demangle (name.c_str(), buf, &size, &status);
   buf[sizeof(buf) - 1] = 0;

   return std::string(buf);
}

rapidjson::Value& flexnnet::NetworkLayerSerializer::encode(rapidjson::Value& _obj, const NetworkLayer& _layer)
{
   _obj.SetObject();

   rapidjson::Value temp_obj;

   _obj.AddMember("name", encodeID(temp_obj, _layer), allocator);

   _obj.AddMember("topology", encodeTopology(temp_obj, _layer), allocator);

   _obj.AddMember("learned_parameters", encodeWeights(temp_obj, _layer), allocator);

   return _obj;
}

rapidjson::Value& flexnnet::NetworkLayerSerializer::encodeID(rapidjson::Value& _obj, const NetworkLayer& _layer)
{
   /*
    * Add layer identity information
    */
   _obj.SetObject();

   std::string layer_name = _layer.name();
   _obj.SetString(layer_name.c_str(), layer_name.size(), allocator);

   return _obj;
}

rapidjson::Value& flexnnet::NetworkLayerSerializer::encodeTopology(rapidjson::Value& _obj, const NetworkLayer& _layer)
{
   /*
    * Add layer dimensions
    */
   _obj.SetObject();

   rapidjson::Value layer_size_obj;
   layer_size_obj.SetInt(_layer.size());
   _obj.AddMember("layer_size", layer_size_obj, allocator);

   rapidjson::Value layer_input_size_obj;
   layer_input_size_obj.SetInt(_layer.input_size());
   _obj.AddMember("layer_input_size", layer_input_size_obj, allocator);

   return _obj;
}

rapidjson::Value& flexnnet::NetworkLayerSerializer::encodeWeights(rapidjson::Value& _obj, const NetworkLayer& _layer)
{
   /*
    * Write weights
    */
   _obj.SetObject();

   rapidjson::Value weights_obj;
   weights_obj.SetArray();

   const Array<double>& weights = _layer.layer_weights.const_weights_ref;
   ArrayToJSONObj(weights_obj, weights);
   _obj.AddMember("weights", weights_obj, allocator);

   return _obj;
}

 flexnnet::NetworkLayerSerializer::NetworkLayerInfo flexnnet::NetworkLayerSerializer::parse (const std::string &_json)
{
   NetworkLayerInfo network_layer_info;

   // Parse json file stream into rapidjson document
   rapidjson::Document document;
   document.Parse(_json.c_str());

   //Save layer identifier information
   network_layer_info.name = document["name"].GetString();

   //Save layer dimension information
   network_layer_info.size = document["topology"]["layer_size"].GetUint();
   network_layer_info.extern_input_size = document["topology"]["layer_input_size"].GetUint();

   //Save layer weights information - resize weights as required
   network_layer_info.weights.resize(network_layer_info.size, network_layer_info.extern_input_size+1);

   const rapidjson::Value &weights_obj = document["learned_parameters"]["weights"].GetArray();
   JSONObjToArray (network_layer_info.weights, weights_obj);

   std::cout << std::endl << "layer size = " << network_layer_info.size << std::endl;

   return network_layer_info;
}

