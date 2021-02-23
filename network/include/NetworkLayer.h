//
// Created by kfedrick on 2/21/21.
//

#ifndef FLEX_NEURALNET_NETWORKLAYER_H_
#define FLEX_NEURALNET_NETWORKLAYER_H_

#include <memory>
#include "flexnnet_networks.h"
#include "BasicLayer.h"
#include "LayerConnRecord.h"

namespace flexnnet
{
   class NetworkLayer
   {
   public:
      NetworkLayer();
      NetworkLayer(bool _is_output);
      NetworkLayer(std::shared_ptr<BasicLayer>& _layer, bool _is_output = false);
      NetworkLayer(std::shared_ptr<BasicLayer>&& _layer, bool _is_output = false);
      ~NetworkLayer();

   public:
      virtual const std::string& name() const;

      /**
       * Returns the size of the layer activity vector.
       * @return
       */
      virtual size_t size() const;

      virtual const std::valarray<double>& value() const;

      // Return true if this is an output basiclayer
      bool is_output_layer(void) const;

      virtual std::shared_ptr<BasicLayer>& layer();

      const std::vector<LayerConnRecord>& get_activation_connections() const;
      const std::vector<LayerConnRecord>& get_backprop_connections() const;
      const std::vector<std::string>& get_external_input_fields() const;

      /**
       * Add a connection to this basiclayer from an external input vector.
       * @param _vec
       */
      void
      add_external_input_field(const std::string& _field);

      /**
       * Add an activation connection to this layer from the layer, _from.
       *
       * @param _from - the name of the layer to send its output
       */
      void
      add_activation_connection(std::shared_ptr<BasicLayer>& _from, LayerConnRecord::ConnectionType _type = LayerConnRecord::Forward);

      /**
       * Add a backprop connection to this layer from the layer, _from.
       *
       * @param _from - the name of the layer to send its output
       */
      void
      add_backprop_connection(std::shared_ptr<BasicLayer>& _from, LayerConnRecord::ConnectionType _type = LayerConnRecord::Forward);

      /**
       * Marshal layer inputs, activate the base layer and return the
       * layer output.
       * @param _externin
       * @return
       */
      virtual const std::valarray<double>& activate(const NNetIO_Typ& _externin);

      /*
       * Protected helper function
       */
   protected:
      /**
       * Marshal layer and external inputs into a single valarray.
       * @param _externin
       * @return
       */
      const std::valarray<double>& marshal_inputs(const NNetIO_Typ& _externin);

      size_t append_virtual_vector(size_t start_ndx, const std::valarray<double>& vec);

   private:
      std::shared_ptr<BasicLayer> basiclayer;
      bool output_layer_flag;

      std::vector<LayerConnRecord> activation_connections;
      std::vector<std::string> external_input_fields;
      std::vector<LayerConnRecord> backprop_connections;

   protected:
      // Local valarray to contain the marshalled input vector for the layer
      std::valarray<double> virtual_input_vector;
   };

   inline std::shared_ptr<BasicLayer>& NetworkLayer::layer()
   {
      return basiclayer;
   }

   inline const std::string& NetworkLayer::name() const
   {
      return basiclayer->name();
   }

   inline size_t NetworkLayer::size() const
   {
      return basiclayer->size();
   }

   inline bool NetworkLayer::is_output_layer(void) const
   {
      return output_layer_flag;
   }

   inline
   const std::valarray<double>& NetworkLayer::value() const
   {
      return basiclayer->const_value;
   }

   inline
   const std::vector<LayerConnRecord>& NetworkLayer::get_activation_connections() const
   {
      return activation_connections;
   }

   inline
   const std::vector<LayerConnRecord>& NetworkLayer::get_backprop_connections() const
   {
      return backprop_connections;
   }

   inline
   const std::vector<std::string>& NetworkLayer::get_external_input_fields() const
   {
      return external_input_fields;
   }
}

#endif //FLEX_NEURALNET_NETWORKLAYER_H_
