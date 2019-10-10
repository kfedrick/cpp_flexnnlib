//
// Created by kfedrick on 5/27/19.
//

#ifndef FLEX_NEURALNET_SOFTMAXSERIALIZER_H_
#define FLEX_NEURALNET_SOFTMAXSERIALIZER_H_

#include <document.h>
#include <SoftMax.h>
#include <LayerSerializer.h>

namespace flexnnet
{
   template<> rapidjson::Value&
   flexnnet::LayerSerializer<SoftMax>::encodeTransferFunction(rapidjson::Value& _obj, const SoftMax& _layer)
   {
      _obj.SetObject();

      /*
       * Add transfer function parameters
       */
      _obj.SetObject();

      rapidjson::Value transfunc_type_obj;
      std::string transfunc_type_str = demangle(typeid(SoftMax).name());
      transfunc_type_obj.SetString(transfunc_type_str.c_str(), transfunc_type_str.size(), allocator);

      _obj.AddMember("type", transfunc_type_obj, allocator);

      rapidjson::Value transfunc_params_obj;
      transfunc_params_obj.SetObject();

      transfunc_params_obj.AddMember("gain", _layer.get_gain(), allocator);
      transfunc_params_obj.AddMember("rescaled", _layer.is_rescaled(), allocator);

      _obj.AddMember("parameters", transfunc_params_obj, allocator);

      return _obj;
   }

   template<> std::shared_ptr<SoftMax> flexnnet::LayerSerializer<SoftMax>::parse(const rapidjson::Value& _obj)
   {
      // First get common network layer information
      BasicLayerInfo network_layer_info = BasicLayerSerializer::parseBasic(_obj);

      std::string type = _obj["transfer_function"]["type"].GetString();
      double gain = _obj["transfer_function"]["parameters"]["gain"].GetDouble();
      bool rescaled = _obj["transfer_function"]["parameters"]["rescaled"].GetBool();

      BasicLayer::NetworkLayerType network_layer_type = BasicLayer::Output;
      if (!network_layer_info.is_output_layer)
         network_layer_type = BasicLayer::Hidden;

      std::shared_ptr<SoftMax> layer_ptr = std::shared_ptr<SoftMax>(new SoftMax(network_layer_info
                                                                                   .size, network_layer_info
                                                                                   .id, network_layer_type));
      layer_ptr->resize_input(network_layer_info.input_size);

      layer_ptr->set_gain(gain);
      layer_ptr->set_rescaled(rescaled);
      layer_ptr->layer_weights.set_weights(network_layer_info.weights);

      return layer_ptr;
   }

}

#endif //FLEX_NEURALNET_SOFTMAXSERIALIZER_H_
