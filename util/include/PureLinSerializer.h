//
// Created by kfedrick on 5/26/19.
//

#ifndef FLEX_NEURALNET_PURELINSERIALIZER_H_
#define FLEX_NEURALNET_PURELINSERIALIZER_H_

#include <document.h>
#include <Layer.h>
#include <PureLinTrans.h>
#include <LayerSerializer.h>

namespace flexnnet
{
   template<> rapidjson::Value& flexnnet::LayerSerializer< Layer<PureLinTrans> >::encodeTransferFunction(rapidjson::Value& _obj, const Layer<PureLinTrans>& _layer)
   {
      _obj.SetObject();

      /*
       * Add transfer function parameters
       */
      _obj.SetObject();

      rapidjson::Value transfunc_type_obj;
      std::string transfunc_type_str = demangle( typeid(PureLinTrans).name() );
      transfunc_type_obj.SetString(transfunc_type_str.c_str(), transfunc_type_str.size(), allocator);

      _obj.AddMember("type", transfunc_type_obj, allocator);

      rapidjson::Value transfunc_params_obj;
      transfunc_params_obj.SetObject();

      transfunc_params_obj.AddMember("gain", _layer.get_gain(), allocator);

      _obj.AddMember("parameters", transfunc_params_obj, allocator);

      return _obj;
   }

   template<> const Layer <PureLinTrans> &flexnnet::LayerSerializer< Layer<PureLinTrans> >::parse (const std::string &_json)
   {
      // First get common network layer information
      NetworkLayerInfo network_layer_info = NetworkLayerSerializer::parse(_json);

      // Parse json file stream into rapidjson document to get transfer function info
      rapidjson::Document document;
      document.Parse(_json.c_str());

      std::string type = document["transfer_function"]["type"].GetString();
      double gain = document["transfer_function"]["parameters"]["gain"].GetDouble();

      static Layer<PureLinTrans> layer(network_layer_info.size, network_layer_info.name);
      //static Layer<PureLinTrans> layer(2, "dummy");

      layer.resize_input_vector (network_layer_info.extern_input_size);
      layer.set_gain(gain);
      layer.layer_weights.set_weights(network_layer_info.weights);

      return layer;
   }

}

#endif //_PURELINSERIALIZER_H_
