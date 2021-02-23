//
// Created by kfedrick on 6/8/19.
//

#ifndef FLEX_NEURALNET_NETWORKLAYERSERIALIZER_H_
#define FLEX_NEURALNET_NETWORKLAYERSERIALIZER_H_

#include <memory>
#include <map>

#include "BasicNeuralNet.h"
#include "ExternalInputRecord.h"
#include "BasicLayerSerializer.h"

namespace flexnnet
{
   class BasicNeuralNetSerializer : public BasicLayerSerializer
   {
   public:

      struct LayerInputInfo
      {
         std::vector<OldLayerConnRecord> layer_conn_info;

         std::set<std::string> external_input_fields;
         std::vector<ExternalInputRecord> external_inputs;
      };

      /**
       * Structure NetworkLayerInfo is used by the static parse method to
       * return information regarding the common network basiclayer components.
       */
      struct BasicNeuralNetInfo
      {
         std::map<std::string, LayerInputInfo> network_topology;
         std::vector<ExternalInputRecord> network_input;
         std::vector<BasicLayerSerializer::BasicLayerInfo> layer_info;

         std::map<std::string, std::shared_ptr<BasicLayer>> layers;
      };

   public:
      static std::shared_ptr<flexnnet::BasicNeuralNet> parse(const std::string& _json);
      static std::string toJson(const BasicNeuralNet& _neural_net);

   public:
      static rapidjson::Value& encode(rapidjson::Value& _obj, const BasicNeuralNet& _neural_net);
      static rapidjson::Value&
      encodeNetworkLayers(rapidjson::Value& _obj, const std::vector<std::shared_ptr<OldNetworkLayer>>& _network_layers);
      static rapidjson::Value&
      encodeLayerTopology(rapidjson::Value& _obj, const std::vector<std::shared_ptr<OldNetworkLayer>>& _network_layers);
      static rapidjson::Value& encodeNetworkConnections(rapidjson::Value& _obj, const OldNetworkLayer& _layer);
      static rapidjson::Value&
      encodeExternalLayerInput(rapidjson::Value& _obj, const std::vector<ExternalInputRecord>& _xinput);
      static rapidjson::Value& encodeConnectionFromLayer(rapidjson::Value& _obj, const OldLayerConnRecord& _conn);
      static rapidjson::Value& encodeNetworkInput(rapidjson::Value& _obj, const Datum& _datum);

   public:
      static std::vector<std::shared_ptr<OldNetworkLayer>>
      parseNetworkTopology(const rapidjson::Value& _obj, std::map<std::string,
                                                                  std::shared_ptr<
                                                                     OldNetworkLayer>>& _layers, const Datum& _network_input);
      static LayerInputInfo parseLayerConnections(const rapidjson::Value& _obj);
      static Datum parseNetworkInput(const rapidjson::Value& _obj);
      static LayerInputInfo parseExternalInput(const rapidjson::Value& _obj);
      static std::string connTypeToString(OldLayerConnRecord::ConnectionType _type);
      static OldLayerConnRecord::ConnectionType StringToConnType(const std::string& _typeStr);
      static std::map<std::string, std::shared_ptr<OldNetworkLayer>> parseNetworkLayers(const rapidjson::Value& _obj);

   };
}

#endif //FLEX_NEURALNET_NETWORKLAYERSERIALIZER_H_
