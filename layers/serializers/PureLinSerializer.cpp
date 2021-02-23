//
// Created by kfedrick on 5/26/19.
//

#include <memory>
#include <document.h>
#include <PureLin.h>
#include <LayerSerializer.h>

using std::shared_ptr;

namespace flexnnet
{
   template<> rapidjson::Value&
   flexnnet::LayerSerializer<PureLin>::encodeTransferFunction(rapidjson::Value& _obj, const PureLin& _layer)
   {
      _obj.SetObject();

      /*
       * Add transfer function parameters
       */

      rapidjson::Value transfunc_type_obj;
      std::string transfunc_type_str = demangle(typeid(PureLin).name());
      transfunc_type_obj.SetString(transfunc_type_str.c_str(), transfunc_type_str.size(), allocator);

      _obj.AddMember("type", transfunc_type_obj, allocator);

      rapidjson::Value transfunc_params_obj;
      transfunc_params_obj.SetObject();

      transfunc_params_obj.AddMember("gain", _layer.get_gain(), allocator);

      _obj.AddMember("parameters", transfunc_params_obj, allocator);

      return _obj;
   }

   template<> std::shared_ptr<PureLin> flexnnet::LayerSerializer<PureLin>::parse(const rapidjson::Value& _obj)
   {
      // First get common network basiclayer information
      BasicLayerInfo network_layer_info = BasicLayerSerializer::parseBasic(_obj);

      std::string type = _obj["transfer_function"]["type"].GetString();
      double gain = _obj["transfer_function"]["parameters"]["gain"].GetDouble();

      OldNetworkLayer::NetworkLayerType network_layer_type = OldNetworkLayer::Output;
      if (!network_layer_info.is_output_layer)
         network_layer_type = OldNetworkLayer::Hidden;

      std::shared_ptr<PureLin> layer_ptr = std::shared_ptr<PureLin>(new PureLin(network_layer_info
                                                                                   .size, network_layer_info
                                                                                   .id, network_layer_type));

      layer_ptr->resize_input(network_layer_info.input_size);

      layer_ptr->set_gain(gain);
      layer_ptr->layer_weights.set(network_layer_info.weights);

      return layer_ptr;
   }
}
