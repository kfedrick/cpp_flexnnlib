//
// Created by kfedrick on 5/26/19.
//

#ifndef FLEX_NEURALNET_LAYERSERIALIZER_H_
#define FLEX_NEURALNET_LAYERSERIALIZER_H_

#include <memory>
#include <document.h>
#include <BasicLayerSerializer.h>

namespace flexnnet
{

   template<class _LayerType> class LayerSerializer : protected BasicLayerSerializer
   {
      friend std::map<std::string, std::shared_ptr<BasicLayer>>
      BasicNeuralNetSerializer::parseNetworkLayers(const rapidjson::Value& _obj);

   public:
      static std::string toJson(const _LayerType& _layer);
      static std::shared_ptr<_LayerType> parse(const std::string& _json);

      static std::shared_ptr<_LayerType> parse(const rapidjson::Value& _obj);
   protected:
      static rapidjson::Value& encodeTransferFunction(rapidjson::Value& _obj, const _LayerType& _layer);

   };

   template<class _LayerType> std::string flexnnet::LayerSerializer<_LayerType>::toJson(const _LayerType& _layer)
   {
      std::string json;

      // First encode common network basic_layer information
      rapidjson::Value network_json_obj;
      BasicLayerSerializer::encodeBasic(network_json_obj, _layer);

      rapidjson::Document doc;
      doc.SetObject();
      doc.CopyFrom(network_json_obj, doc.GetAllocator());

      rapidjson::Value transfunc_json_obj;
      LayerSerializer<_LayerType>::encodeTransferFunction(transfunc_json_obj, _layer);
      doc.AddMember("transfer_function", transfunc_json_obj, doc.GetAllocator());

      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> jsonWriter(buffer);
      jsonWriter.SetMaxDecimalPlaces(7);

      doc.Accept(jsonWriter);
      std::string json_string = buffer.GetString();

      return json_string;
   }

   template<class _LayerType>
   std::shared_ptr<_LayerType> flexnnet::LayerSerializer<_LayerType>::parse(const std::string& _json)
   {
      // Parse json file stream into rapidjson document to get transfer function info
      rapidjson::Document document;
      document.Parse(_json.c_str());

      rapidjson::Value _obj = document.GetObject();
      return flexnnet::LayerSerializer<_LayerType>::parse(_obj);
   }
}

#endif //FLEX_NEURALNET_LAYERSERIALIZER_H_
