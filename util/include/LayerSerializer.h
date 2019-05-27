//
// Created by kfedrick on 5/26/19.
//

#ifndef FLEX_NEURALNET_LAYERSERIALIZER_H_
#define FLEX_NEURALNET_LAYERSERIALIZER_H_

#include <document.h>
#include <NetworkLayerSerializer.h>

namespace flexnnet
{
   template<class _LayerType> class LayerSerializer : protected NetworkLayerSerializer
   {
   public:
      static const _LayerType &parse (const std::string &_json);
      static std::string toJSON(const _LayerType& _layer);

   protected:
      static rapidjson::Value& encodeTransferFunction(rapidjson::Value& _obj, const _LayerType& _layer);

   };

   template<class _LayerType> std::string flexnnet::LayerSerializer<_LayerType>::toJSON(const _LayerType& _layer)
   {
      std::string json;

      // First encode common network layer information
      rapidjson::Value network_json_obj;
      NetworkLayerSerializer::encode(network_json_obj, _layer);

      rapidjson::Document doc;
      doc.SetObject();
      doc.CopyFrom(network_json_obj, doc.GetAllocator());

      rapidjson::Value transfunc_json_obj;
      encodeTransferFunction (transfunc_json_obj, _layer);
      doc.AddMember("transfer_function", transfunc_json_obj, doc.GetAllocator());

      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> jsonWriter(buffer);
      jsonWriter.SetMaxDecimalPlaces (7);

      doc.Accept(jsonWriter);
      std::string json_string = buffer.GetString();

      return json_string;
   }
}

#endif //FLEX_NEURALNET_LAYERSERIALIZER_H_
