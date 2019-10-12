//
// Created by kfedrick on 5/26/19.
//

#ifndef FLEX_NEURALNET_BASICLAYERSERIALIZER_H_
#define FLEX_NEURALNET_BASICLAYERSERIALIZER_H_

#include <document.h>
#include "BasicLayer.h"
#include "NetworkLayer.h"
#include "JSONEncoder.h"

namespace flexnnet
{
   class BasicLayerSerializer : protected JSONEncoder
   {
   protected:

      /**
       * Structure BasicLayerInfo is used by the static parse method to
       * return information regarding the common network layer components.
       */
      struct BasicLayerInfo
      {
         // The layer id
         std::string id;

         // The length of the layer output vector
         size_t size;

         // The length of the input vector
         size_t input_size;

         bool is_output_layer;

         // The network weights
         Array2D<double> weights;
      };

   protected:
      static BasicLayerInfo parseBasic(const std::string& _json);
      static BasicLayerInfo parseBasic(const rapidjson::Value& _obj);
      static rapidjson::Value& encodeBasic(rapidjson::Value& _obj, const BasicLayer& _layer);
      static std::string demangle(const std::string& name);

   private:
      static rapidjson::Value& encodeID(rapidjson::Value& _obj, const BasicLayer& _layer);
      static rapidjson::Value& encodeDimensions(rapidjson::Value& _obj, const BasicLayer& _layer);
      static rapidjson::Value& encodeWeights(rapidjson::Value& _obj, const BasicLayer& _layer);

   };
}

#endif //FLEX_NEURALNET_BASICLAYERSERIALIZER_H_
