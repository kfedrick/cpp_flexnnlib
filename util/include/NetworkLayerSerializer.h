//
// Created by kfedrick on 5/26/19.
//

#ifndef FLEX_NEURALNET_NETWORKLAYERSERIALIZER_H_
#define FLEX_NEURALNET_NETWORKLAYERSERIALIZER_H_

#include <document.h>
#include <NetworkLayer.h>
#include <JSONEncoder.h>

namespace flexnnet
{
   class NetworkLayerSerializer : protected JSONEncoder
   {
   protected:

      /**
       * Structure NetworkLayerInfo is used by the static parse method to
       * return information regarding the common network layer components.
       */
      struct NetworkLayerInfo
      {
         std::string name;
         unsigned int size;
         unsigned int extern_input_size;
         Array<double> weights;
      };

   public:
      NetworkLayerSerializer();
      ~NetworkLayerSerializer();

   protected:
      static NetworkLayerInfo parse (const std::string &_json);
      static rapidjson::Value& encode(rapidjson::Value& _obj, const NetworkLayer& _layer);
      std::string toString();
      static std::string demangle(const std::string& name);


   private:
      static rapidjson::Value& encodeID(rapidjson::Value& _obj, const NetworkLayer& _layer);
      static rapidjson::Value& encodeTopology(rapidjson::Value& _obj, const NetworkLayer& _layer);
      static rapidjson::Value& encodeWeights(rapidjson::Value& _obj, const NetworkLayer& _layer);

   };
}

#endif //_NETWORKLAYERSERIALIZER_H_
