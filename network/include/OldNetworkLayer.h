//
// Created by kfedrick on 6/5/19.
//

#ifndef FLEX_NEURALNET_OLDLAYERCONNECTOR_H_
#define FLEX_NEURALNET_OLDLAYERCONNECTOR_H_

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <valarray>

#include "OldDatum.h"
#include "BasicLayer.h"
#include "ExternalInputRecord.h"
#include "LayerInput.h"

namespace flexnnet
{
   class OldNetworkLayer : public BasicLayer, public LayerInput
   {

   public:
      enum NetworkLayerType
      {
         Output, Hidden
      };

   protected:
      /* ********************************************************************
       * Constructors, destructors
       */

      OldNetworkLayer(size_t _sz, const std::string& _name, NetworkLayerType _type = Output);

   public:

      ~OldNetworkLayer();

   public:
      // Return true if this is an output basic_layer
      bool is_output_layer(void) const;

      /* ******************************************************************
       * Public member functions to connect layers and external inputs.
       */
      size_t add_connection(BasicLayer& _layer, OldLayerConnRecord::ConnectionType _type);
      size_t add_external_input(const OldDatum& _xdatum, const std::set<std::string>& _indexSet);

   private:
      const NetworkLayerType network_layer_type;

   };

   inline bool OldNetworkLayer::is_output_layer(void) const
   {
      return (network_layer_type == Output);
   }

   inline size_t OldNetworkLayer::add_connection(BasicLayer& _layer, OldLayerConnRecord::ConnectionType _type)
   {
      LayerInput::add_connection(_layer, _type);

      // Resize input for basic basic_layer
      resize_input(virtual_input_size());
   }

   inline size_t OldNetworkLayer::add_external_input(const OldDatum& _xdatum, const std::set<std::string>& _indexSet)
   {
      LayerInput::add_external_input(_xdatum, _indexSet);

      // Resize input for basic basic_layer
      resize_input(virtual_input_size());

      return virtual_input_size();
   }

}

#endif //FLEX_NEURALNET_LAYERCONNECTOR_H_
