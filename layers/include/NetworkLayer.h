//
// Created by kfedrick on 6/5/19.
//

#ifndef FLEX_NEURALNET_LAYERCONNECTOR_H_
#define FLEX_NEURALNET_LAYERCONNECTOR_H_

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <valarray>

#include "Datum.h"
#include "BasicLayer.h"
#include "ExternalInputRecord.h"
#include "LayerInput.h"

namespace flexnnet
{
   class NetworkLayer : public BasicLayer, public LayerInput
   {

   protected:
      /* ********************************************************************
       * Constructors, destructors
       */

      NetworkLayer(size_t _sz, const std::string& _name, NetworkLayerType _type = Output);

   public:

      ~NetworkLayer();

   public:
      /* ******************************************************************
       * Public member functions to connect layers and external inputs.
       */
      size_t add_connection(BasicLayer& _layer, LayerConnRecord::ConnectionType _type);
      size_t add_external_input(const Datum& _xdatum, const std::set<std::string>& _indexSet);
   };

   inline size_t NetworkLayer::add_connection(BasicLayer& _layer, LayerConnRecord::ConnectionType _type)
   {
      LayerInput::add_connection(_layer, _type);

      // Resize input for basic layer
      resize_input(virtual_input_size());
   }

   inline size_t NetworkLayer::add_external_input(const Datum& _xdatum, const std::set<std::string>& _indexSet)
   {
      LayerInput::add_external_input(_xdatum, _indexSet);

      // Resize input for basic layer
      resize_input(virtual_input_size());

      return virtual_input_size();
   }

}

#endif //FLEX_NEURALNET_LAYERCONNECTOR_H_
