//
// Created by kfedrick on 5/26/19.
//

#ifndef FLEX_NEURALNET_LOGSIGSERIALIZER_H_
#define FLEX_NEURALNET_LOGSIGSERIALIZER_H_

#include <document.h>
#include <LogSig.h>
#include <LayerSerializer.h>

namespace flexnnet
{
   template<> rapidjson::Value&
   flexnnet::LayerSerializer<LogSig>::encodeTransferFunction(rapidjson::Value& _obj, const LogSig& _layer)
   {
      _obj.SetObject();

      /*
       * Add transfer function parameters
       */
      _obj.SetObject();

      rapidjson::Value transfunc_type_obj;
      std::string transfunc_type_str = demangle(typeid(LogSig).name());
      transfunc_type_obj.SetString(transfunc_type_str.c_str(), transfunc_type_str.size(), allocator);

      _obj.AddMember("type", transfunc_type_obj, allocator);

      rapidjson::Value transfunc_params_obj;
      transfunc_params_obj.SetObject();

      transfunc_params_obj.AddMember("gain", _layer.get_gain(), allocator);

      _obj.AddMember("parameters", transfunc_params_obj, allocator);

      return _obj;
   }

   template<> std::shared_ptr<LogSig> flexnnet::LayerSerializer<LogSig>::parse(const rapidjson::Value& _obj)
   {
      // First get common network basic_layer information
      BasicLayerInfo network_layer_info = BasicLayerSerializer::parseBasic(_obj);

      std::string type = _obj["transfer_function"]["type"].GetString();
      double gain = _obj["transfer_function"]["parameters"]["gain"].GetDouble();

      OldNetworkLayer::NetworkLayerType network_layer_type = OldNetworkLayer::Output;
      if (!network_layer_info.is_output_layer)
         network_layer_type = OldNetworkLayer::Hidden;

      std::shared_ptr<LogSig> layer_ptr = std::shared_ptr<LogSig>(new LogSig(network_layer_info.size, network_layer_info
         .id));
      layer_ptr->resize_input(network_layer_info.input_size);

      layer_ptr->set_gain(gain);
      layer_ptr->layer_weights.set(network_layer_info.weights);

      return layer_ptr;
   }
}

#endif //FLEX_NEURALNET_LOGSIGSERIALIZER_H_
