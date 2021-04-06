//
// Created by kfedrick on 6/5/19.
//

#ifndef FLEX_NEURALNET_LAYERCONNRECORDD_H_
#define FLEX_NEURALNET_LAYERCONNRECORDD_H_

#include <stddef.h>
#include "BasicLayer.h"

namespace flexnnet
{
   class OldLayerConnRecord
   {
   public:
      enum ConnectionType
      {
         Forward = 0, Recurrent = 1, Lateral = 2
      };

   public:
      OldLayerConnRecord() : connection_type(Forward), input_layer_size(0), input_layer(0)
      {}
      OldLayerConnRecord(flexnnet::BasicLayer* _from, ConnectionType _type);
      OldLayerConnRecord(size_t _sz, ConnectionType _type);
      OldLayerConnRecord(std::string _id, size_t _sz, ConnectionType _type);

      ConnectionType get_connection_type() const;

      bool is_recurrent() const;

      size_t size() const;
      BasicLayer& get_input_layer() const;
      const std::string& get_input_layer_id() const;

   private:
      std::string input_layer_id;
      BasicLayer* input_layer;
      ConnectionType connection_type;
      size_t input_layer_size;
   };

   inline OldLayerConnRecord::ConnectionType OldLayerConnRecord::get_connection_type() const
   {
      return connection_type;
   }

   inline bool OldLayerConnRecord::is_recurrent() const
   {
      return !(connection_type == Forward);
   }

   inline size_t OldLayerConnRecord::size() const
   {
      return input_layer_size;
   }

   inline BasicLayer& OldLayerConnRecord::get_input_layer() const
   {
      return *input_layer;
   }

   inline const std::string& OldLayerConnRecord::get_input_layer_id() const
   {
      return input_layer_id;
   }
}

#endif //FLEX_NEURALNET_LAYERCONNRECORD_H_
