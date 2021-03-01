//
// Created by kfedrick on 2/28/21.
//

#ifndef FLEX_NEURALNET_NETWORKLAYER_H_
#define FLEX_NEURALNET_NETWORKLAYER_H_

#include <memory>
#include "flexnnet.h"
#include "BasicLayer.h"
#include "LayerConnRecord.h"

namespace flexnnet
{
   class NetworkTopology;

   class NetworkLayer
   {
      friend class NetworkTopology;

   public:
      NetworkLayer();
      NetworkLayer(bool _is_output);
      NetworkLayer(const std::shared_ptr<BasicLayer>& _layer, bool _is_output = false);
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

      // Return true if this is an output basic_layer
      virtual bool is_output_layer(void) const;

      LayerWeights& weights();

      const Array2D<double>& dAdN(void) const;

      const Array2D<double>& dNdW(void) const;

      const Array2D<double>& dNdI(void) const;

      const std::vector<LayerConnRecord>& get_activation_connections() const;
      const std::vector<LayerConnRecord>& get_backprop_connections() const;
      const std::vector<ExternalInputRecord>& get_external_input_fields() const;

   protected:
      virtual std::shared_ptr<BasicLayer>& basiclayer();

      /**
       * Marshal layer and external inputs into a single valarray.
       * @param _externin
       * @return
       */
      const std::valarray<double>& marshal_inputs(const NNetIO_Typ& _externin);

      size_t append_virtual_vector(size_t start_ndx, const std::valarray<double>& vec);

   /* **********************************************************
    * Private configurator functions
    */
   private:
      void clear(void);

      /**
       * Add a connection to this basic_layer from an external input vector.
       * @param _vec
       */
      void
      add_external_input_field(const std::string& _field, size_t _sz);

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

   protected:
      const std::valarray<double>& virtual_input_vector_const_ref = virtual_input_vector;

   /* ***********************************************************
    * Private data members
    */
   private:
      std::shared_ptr<BasicLayer> basic_layer;
      bool output_layer_flag;

      std::vector<LayerConnRecord> activation_connections;
      std::vector<ExternalInputRecord> external_input_fields;
      std::vector<LayerConnRecord> backprop_connections;

      // Local valarray to contain the marshalled input vector for the layer
      std::valarray<double> virtual_input_vector;
   };

   inline std::shared_ptr<BasicLayer>& NetworkLayer::basiclayer()
   {
      return basic_layer;
   }

   inline const std::string& NetworkLayer::name() const
   {
      return basic_layer->name();
   }

   inline size_t NetworkLayer::size() const
   {
      return basic_layer->size();
   }

   inline bool NetworkLayer::is_output_layer(void) const
   {
      return output_layer_flag;
   }

   inline
   const std::valarray<double>& NetworkLayer::value() const
   {
      return basic_layer->const_value;
   }

   inline
   LayerWeights& NetworkLayer::weights()
   {
      return basic_layer->layer_weights;
   }

   inline
   const Array2D<double>& NetworkLayer::dAdN(void) const
   {
      return basic_layer->get_dAdN();
   }

   inline
   const Array2D<double>& NetworkLayer::dNdW(void) const
   {
      return basic_layer->get_dNdW();
   }

   inline
   const Array2D<double>& NetworkLayer::dNdI(void) const
   {
      return basic_layer->get_dNdI();
   }

   inline
   void NetworkLayer::clear(void)
   {
      basic_layer.reset();
      virtual_input_vector.resize(0);
      activation_connections.clear();
      external_input_fields.clear();
      backprop_connections.clear();
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
   const std::vector<ExternalInputRecord>& NetworkLayer::get_external_input_fields() const
   {
      return external_input_fields;
   }
}

#endif //FLEX_NEURALNET_NETWORKLAYER_H_
