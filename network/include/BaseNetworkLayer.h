//
// Created by kfedrick on 4/10/21.
//

#ifndef FLEX_NEURALNET_NETWORKLAYERBUILDER_H_
#define FLEX_NEURALNET_NETWORKLAYERBUILDER_H_

#include <string>
#include <vector>
#include <memory>

#include <flexnnet.h>
#include "ExternalInputRecord.h"
#include "LayerConnRecord.h"

namespace flexnnet
{
   class BaseNetworkLayer
   {
   public:
      BaseNetworkLayer();
      BaseNetworkLayer(const BaseNetworkLayer& _layer);
      virtual ~BaseNetworkLayer();

      virtual const std::string& name(void) const = 0;

      void clear();

      /**
       * Add a connection to this network layer from an external input vector.
       * @param _vec
       */
      void
      add_external_input_field(const std::string& _field, size_t _sz);

      void add_connection(const std::string& _cid, const std::shared_ptr<NetworkLayer>& _const, LayerConnRecord::ConnectionType _type);

   protected:
      unsigned int calc_input_size(void);
      virtual void set_input_size(size_t _rawin_sz) = 0;

   protected:
      std::vector<LayerConnRecord> activation_connections;
      std::vector<ExternalInputRecord> external_input_fields;
      std::vector<LayerConnRecord> backprop_connections;

      // Local data member to hold scattered partial errors with
      // respect to the input to be back-propagated to layers that
      // feed activity to this layer.
      flexnnet::ValarrMap input_error_map;
      flexnnet::ValarrMap external_input_error_map;
   };

} // end namespace flexnnet

#endif //_NETWORKLAYERBUILDER_H_
