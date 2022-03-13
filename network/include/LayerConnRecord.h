//
// Created by kfedrick on 4/9/21.
//

#ifndef FLEX_NEURALNET_LAYERCONNRECORD_H_
#define FLEX_NEURALNET_LAYERCONNRECORD_H_

#include <stddef.h>
#include <memory>

namespace flexnnet
{
   class NetworkLayer;

   class LayerConnRecord
   {

   public:
      enum ConnectionType
      {
         Forward = 0, Recurrent = 1, Lateral = 2
      };

   public:
      LayerConnRecord();
      LayerConnRecord(const LayerConnRecord& _crec);
      LayerConnRecord(const std::shared_ptr<NetworkLayer>& _from_layer, ConnectionType _type);
      ~LayerConnRecord();

      NetworkLayer&
      layer();

      const NetworkLayer&
      layer() const;

      bool
      is_recurrent() const;

      ConnectionType
      get_connection_type() const;

   private:
      std::shared_ptr<NetworkLayer> from_layer;
      ConnectionType connection_type;
   };

   inline
   LayerConnRecord::LayerConnRecord()
   {}

   inline
   LayerConnRecord::LayerConnRecord(const LayerConnRecord& _crec) : connection_type(_crec.connection_type)
   {
      from_layer = _crec.from_layer;
   }

   inline
   LayerConnRecord::LayerConnRecord(const std::shared_ptr<NetworkLayer>& _from_layer, ConnectionType _type)
      : connection_type(_type)
   {
      from_layer = _from_layer;
   }

   inline
   LayerConnRecord::~LayerConnRecord()
   {}

   inline
   NetworkLayer&
   LayerConnRecord::layer()
   {
      return *from_layer;
   }

   inline
   const NetworkLayer&
   LayerConnRecord::layer() const
   {
      return *from_layer;
   }

   inline bool
   LayerConnRecord::is_recurrent() const
   {
      return !(connection_type == Forward);
   }

   inline LayerConnRecord::ConnectionType
   LayerConnRecord::get_connection_type() const
   {
      return connection_type;
   }
}

#endif //FLEX_NEURALNET_LAYERCONNRECORD_H_
