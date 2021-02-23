//
// Created by kfedrick on 5/26/19.
//

#ifndef FLEX_NEURALNET_RADBASSERIALIZER_H_
#define FLEX_NEURALNET_RADBASSERIALIZER_H_

#include <document.h>
#include <RadBas.h>
#include <LayerSerializer.h>

namespace flexnnet
{
   template<> rapidjson::Value&
   flexnnet::LayerSerializer<RadBas>::encodeTransferFunction(rapidjson::Value& _obj, const RadBas& _layer)
   {
      _obj.SetObject();

      /*
       * Add transfer function parameters
       */
      _obj.SetObject();

      rapidjson::Value transfunc_type_obj;
      std::string transfunc_type_str = demangle(typeid(RadBas).name());
      transfunc_type_obj.SetString(transfunc_type_str.c_str(), transfunc_type_str.size(), allocator);

      _obj.AddMember("type", transfunc_type_obj, allocator);

      rapidjson::Value transfunc_params_obj;
      transfunc_params_obj.SetObject();

      transfunc_params_obj.AddMember("rescaled", _layer.is_rescaled(), allocator);

      _obj.AddMember("parameters", transfunc_params_obj, allocator);

      return _obj;
   }

   template<> std::shared_ptr<RadBas> flexnnet::LayerSerializer<RadBas>::parse(const rapidjson::Value& _obj)
   {
      // First get common network basiclayer information
      BasicLayerInfo network_layer_info = BasicLayerSerializer::parseBasic(_obj);

      std::string type = _obj["transfer_function"]["type"].GetString();
      bool rescaled = _obj["transfer_function"]["parameters"]["rescaled"].GetBool();

      OldNetworkLayer::NetworkLayerType network_layer_type = OldNetworkLayer::Output;
      if (!network_layer_info.is_output_layer)
         network_layer_type = OldNetworkLayer::Hidden;

      std::shared_ptr<RadBas> layer_ptr = std::shared_ptr<RadBas>(new RadBas(network_layer_info.size, network_layer_info
         .id, network_layer_type));
      layer_ptr->resize_input(network_layer_info.input_size);

      layer_ptr->set_rescaled(rescaled);
      layer_ptr->layer_weights.set(network_layer_info.weights);

      return layer_ptr;
   }

}

#endif //FLEX_NEURALNET_RADBASSERIALIZER_H_
