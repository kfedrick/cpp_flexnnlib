//
// Created by kfedrick on 5/26/19.
//

#ifndef FLEX_NEURALNET_RADIALBASISSERIALIZER_H_
#define FLEX_NEURALNET_RADIALBASISSERIALIZER_H_

#include <document.h>
#include <Layer.h>
#include <RadialBasisTrans.h>
#include <LayerSerializer.h>

namespace flexnnet
{
   template<> rapidjson::Value& flexnnet::LayerSerializer< Layer<RadialBasisTrans> >::encodeTransferFunction(rapidjson::Value& _obj, const Layer<RadialBasisTrans>& _layer)
   {
      _obj.SetObject();

      /*
       * Add transfer function parameters
       */
      _obj.SetObject();

      rapidjson::Value transfunc_type_obj;
      std::string transfunc_type_str = demangle( typeid(RadialBasisTrans).name() );
      transfunc_type_obj.SetString(transfunc_type_str.c_str(), transfunc_type_str.size(), allocator);

      _obj.AddMember("type", transfunc_type_obj, allocator);

      rapidjson::Value transfunc_params_obj;
      transfunc_params_obj.SetObject();

      _obj.AddMember("parameters", transfunc_params_obj, allocator);

      return _obj;
   }

   template<> const Layer <RadialBasisTrans> &flexnnet::LayerSerializer< Layer<RadialBasisTrans> >::parse (const std::string &_json)
   {
      // First get common network layer information
      NetworkLayerInfo network_layer_info = NetworkLayerSerializer::parse(_json);

      // Parse json file stream into rapidjson document to get transfer function info
      rapidjson::Document document;
      document.Parse(_json.c_str());

      std::string type = document["transfer_function"]["type"].GetString();
      double gain = document["transfer_function"]["parameters"]["gain"].GetDouble();

      static Layer<RadialBasisTrans> layer(network_layer_info.size, network_layer_info.name);

      layer.resize_input_vector (network_layer_info.extern_input_size);
      layer.layer_weights.set_weights(network_layer_info.weights);

      return layer;
   }

}


#endif //FLEX_NEURALNET_RADIALBASISSERIALIZER_H_
