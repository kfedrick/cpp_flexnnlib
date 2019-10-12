//
// Created by kfedrick on 9/25/19.
//

#ifndef FLEX_NEURALNET_NEURALNETSERIALIZER_H_
#define FLEX_NEURALNET_NEURALNETSERIALIZER_H_

#include "NeuralNet.h"
#include "BasicNeuralNetSerializer.h"

namespace flexnnet
{
   template<class _InType, class _OutType>
   class NeuralNetSerializer
   {
   public:
      static std::shared_ptr<NeuralNet<_InType, _OutType>> parse(const std::string& _json);
      static std::string toJson(const NeuralNet<_InType, _OutType>& _neural_net);
   };

   template<class _InType, class _OutType>
   std::shared_ptr<NeuralNet<_InType, _OutType>> NeuralNetSerializer<_InType, _OutType>::parse(const std::string& _json)
   {
      rapidjson::Document netdoc;
      netdoc.Parse(_json.c_str());

      Datum network_input = BasicNeuralNetSerializer::parseNetworkInput(netdoc["network_input"].GetArray());

      std::map<std::string, std::shared_ptr<NetworkLayer>> layers;
      layers = BasicNeuralNetSerializer::parseNetworkLayers(netdoc["network_layers"].GetArray());

      std::vector<std::shared_ptr<NetworkLayer>> network_layers;
      network_layers = BasicNeuralNetSerializer::parseNetworkTopology(netdoc["layer_topology"]
                                                                         .GetArray(), layers, network_input);

      std::shared_ptr<NeuralNet<_InType, _OutType>> net = std::shared_ptr<NeuralNet<_InType, _OutType>>(new NeuralNet<
         _InType,
         _OutType>(network_layers, false));

      return net;
   }

   template<class _InType, class _OutType>
   std::string NeuralNetSerializer<_InType, _OutType>::toJson(const NeuralNet<_InType, _OutType>& _neural_net)
   {

   }
}

#endif //FLEX_NEURALNET_NEURALNETSERIALIZER_H_
