//
// Created by kfedrick on 2/21/21.
//

#ifndef FLEX_NEURALNET_LAYERCONNRECORD_H_
#define FLEX_NEURALNET_LAYERCONNRECORD_H_

#include <stddef.h>
#include <memory>
#include "BasicLayer.h"

namespace flexnnet
{
   class LayerConnRecord
   {
   public:
      enum ConnectionType
      {
         Forward = 0, Recurrent = 1, Lateral = 2
      };

   public:
      LayerConnRecord();
      LayerConnRecord(std::shared_ptr<BasicLayer> _from_layer, ConnectionType _type);
      ~LayerConnRecord();

      BasicLayer& layer();
      const BasicLayer& layer() const;
      bool is_recurrent() const;
      ConnectionType get_connection_type() const;

   private:
      std::shared_ptr<BasicLayer> from_layer;
      ConnectionType connection_type;
   };

   inline
   BasicLayer& LayerConnRecord::layer()
   {
      return *from_layer;
   }

   inline
   const BasicLayer& LayerConnRecord::layer() const
   {
      return *from_layer;
   }

   inline bool LayerConnRecord::is_recurrent() const
   {
      return !(connection_type == Forward);
   }

   inline LayerConnRecord::ConnectionType LayerConnRecord::get_connection_type() const
   {
      return connection_type;
   }
}

#endif //FLEX_NEURALNET_LAYERCONNRECORD_H_
