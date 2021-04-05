//
// Created by kfedrick on 8/25/19.
//

#ifndef FLEX_NEURALNET_LAYERINPUT_H_
#define FLEX_NEURALNET_LAYERINPUT_H_

#include <stddef.h>
#include <vector>
#include <memory>

#include "OldDatum.h"
#include "BasicLayer.h"
#include "ExternalInputRecord.h"
#include "OldLayerConnRecord.h"

namespace flexnnet
{
   class LayerInput
   {
   public:
      size_t virtual_input_size(void) const;
      const std::vector<OldLayerConnRecord>& get_input_connections() const;
      std::vector<OldLayerConnRecord>& get_input_connections();
      const std::vector<ExternalInputRecord>& get_external_inputs() const;

   public:
      /* ******************************************************************
       * Public member functions to connect layers and external inputs.
       */
      virtual size_t add_connection(BasicLayer& _layer, OldLayerConnRecord::ConnectionType _type);
      virtual size_t add_external_input(const OldDatum& _xdatum, const std::set<std::string>& _indexSet);
      const std::valarray<double>& coelesce_input(const OldDatum& _xdatum);
      void backprop_scatter(const std::valarray<double> _errorv);

   private:
      size_t append_virtual_vector(size_t start_ndx, const std::valarray<double>& vec);

   private:

      std::vector<ExternalInputRecord> external_inputs;

      /*
       * Contains on ordered list of references to the network layers that provide input this
       * network basic_layer.
       */
      std::vector<OldLayerConnRecord> input_layers;

      std::set<std::string> input_layer_names;

      /*
       * Local valarray to contain the virtual input vector from the input layers
       */
      std::valarray<double> virtual_input_vector;

      /*
       * Local valarray to hold the backpropogated error vector for each input basic_layer
       */
      std::vector<std::valarray<double> > backprop_error_vector;
   };

   inline size_t LayerInput::virtual_input_size(void) const
   {
      return virtual_input_vector.size();
   }

   inline const std::vector<OldLayerConnRecord>& LayerInput::get_input_connections() const
   {
      return input_layers;
   }

   inline std::vector<OldLayerConnRecord>& LayerInput::get_input_connections()
   {
      return input_layers;
   }

   inline const std::vector<ExternalInputRecord>& LayerInput::get_external_inputs() const
   {
      return external_inputs;
   }
}

#endif //FLEX_NEURALNET_LAYERINPUT_H_
