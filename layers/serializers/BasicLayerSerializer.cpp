//
// Created by kfedrick on 5/26/19.
//
#include <typeinfo>
#include <cxxabi.h>
#include <iostream>
#include "BasicLayerSerializer.h"

using flexnnet::BasicLayerSerializer;

std::string flexnnet::BasicLayerSerializer::demangle(const std::string& name)
{
   static char buf[2048];
   size_t size = sizeof(buf);
   int status;

   char* res = abi::__cxa_demangle(name.c_str(), buf, &size, &status);
   buf[sizeof(buf) - 1] = 0;

   return std::string(buf);
}

rapidjson::Value& flexnnet::BasicLayerSerializer::encodeBasic(rapidjson::Value& _obj, const BasicLayer& _layer)
{
   _obj.SetObject();

   rapidjson::Value temp_obj;

   _obj.AddMember("id", encodeID(temp_obj, _layer), allocator);

   _obj.AddMember("dimensions", encodeDimensions(temp_obj, _layer), allocator);

   _obj.AddMember("learned_parameters", encodeWeights(temp_obj, _layer), allocator);

   return _obj;
}

rapidjson::Value& flexnnet::BasicLayerSerializer::encodeID(rapidjson::Value& _obj, const BasicLayer& _layer)
{
   /*
    * Add basic_layer identity information
    */
   _obj.SetObject();

   std::string layer_name = _layer.name();
   _obj.SetString(layer_name.c_str(), layer_name.size(), allocator);

   return _obj;
}

rapidjson::Value& flexnnet::BasicLayerSerializer::encodeDimensions(rapidjson::Value& _obj, const BasicLayer& _layer)
{
   /*
    * Add basic_layer dimensions
    */
   _obj.SetObject();

   rapidjson::Value layer_size_obj;
   layer_size_obj.SetUint(_layer.size());
   _obj.AddMember("layer_size", layer_size_obj, allocator);

   rapidjson::Value layer_input_size_obj;
   layer_input_size_obj.SetUint(_layer.input_size());
   _obj.AddMember("layer_input_size", layer_input_size_obj, allocator);

   return _obj;
}

rapidjson::Value& flexnnet::BasicLayerSerializer::encodeWeights(rapidjson::Value& _obj, const BasicLayer& _layer)
{
   /*
    * Write weights
    */
   _obj.SetObject();

   rapidjson::Value weights_obj;
   weights_obj.SetArray();

   const Array2D<double>& weights = _layer.layer_weights.const_weights_ref;
   ArrayToJSONObj(weights_obj, weights);
   _obj.AddMember("weights", weights_obj, allocator);

   return _obj;
}

flexnnet::BasicLayerSerializer::BasicLayerInfo flexnnet::BasicLayerSerializer::parseBasic(const std::string& _json)
{
   BasicLayerInfo network_layer_info;

   // Parse json file stream into rapidjson document
   rapidjson::Document document;
   document.Parse(_json.c_str());

   rapidjson::Value _obj = document.GetObject();
   return parseBasic(_obj);
}

flexnnet::BasicLayerSerializer::BasicLayerInfo flexnnet::BasicLayerSerializer::parseBasic(const rapidjson::Value& _obj)
{
   BasicLayerInfo network_layer_info;

   // Save basic_layer identifier information
   network_layer_info.id = _obj["id"].GetString();

   // Save flag indicating this is an output basic_layer
   network_layer_info.is_output_layer = _obj["is_output_layer"].GetBool();

   // Save basic_layer dimension information
   network_layer_info.size = _obj["dimensions"]["layer_size"].GetUint();
   network_layer_info.input_size = _obj["dimensions"]["layer_input_size"].GetUint();

   // Save basic_layer weights information - resize weights as required
   network_layer_info.weights.resize(network_layer_info.size, network_layer_info.input_size + 1);

   const rapidjson::Value& weights_obj = _obj["learned_parameters"]["weights"];
   JSONObjToArray(network_layer_info.weights, weights_obj);

   return network_layer_info;
}

