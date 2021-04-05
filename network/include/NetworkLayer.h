//
// Created by kfedrick on 2/28/21.
//

#ifndef FLEX_NEURALNET_NETWORKLAYER_H_
#define FLEX_NEURALNET_NETWORKLAYER_H_

#include <memory>
#include <iostream>
#include "flexnnet.h"
#include "BasicLayer.h"
#include "LayerConnRecord.h"
#include "ExternalInputRecord.h"

namespace flexnnet
{
   class NetworkTopology;
   class BaseNeuralNet;

   class NetworkLayer
   {
      friend class NetworkTopology;
      friend class BaseNeuralNet;

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

      virtual const ValarrMap & input_value_map() const;

      void set_weight_initializer(std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)>& _func);

      /**
       * Marshal layer inputs, activate the base layer and return the
       * layer output.
       * @param _externin
       * @return
       */
      virtual const std::valarray<double>& activate(const ValarrMap& _externin);

      virtual const std::valarray<double>& backprop(const ValarrMap& _externerror);

      // Return true if this is an output basic_layer
      virtual bool is_output_layer(void) const;

      void set_biases(double _val);

      void set_biases(const std::valarray<double>& _biases);

      void set_weights(double _val);

      void set_weights(const Array2D<double>& _weights);

      void initialize_weights(void);

      LayerWeights& weights();

      const LayerWeights& weights() const;

      const LayerState& layer_state(void) const;

      const Array2D<double>& dy_dnet(void) const;

      const Array2D<double>& dnet_dw(void) const;

      const Array2D<double>& dnet_dx(void) const;

      const Array2D<double>& dE_dw(void) const;

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
      const std::valarray<double>& concat_inputs(const ValarrMap& _externin);
      const ValarrMap& marshal_inputs(const ValarrMap& _externin);

      /**
       * Marshal the external and back-propagated errors to calculate
       * the cumulative external error vector for this layer.
       *
       * @param _externerr
       * @return
       */
      const std::valarray<double>& gather_error(const ValarrMap& _externerr);

      void scatter_input_error(const std::valarray<double>& _input_errorv);

      size_t append_virtual_vector(size_t start_ndx, const std::valarray<double>& vec);

   /* **********************************************************
    * Private configurator functions
    */
   private:
      void clear(void);

      std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)> _weight_initializer;

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
      add_activation_connection(const std::shared_ptr<NetworkLayer>& _from, LayerConnRecord::ConnectionType _type = LayerConnRecord::Forward);

      /**
       * Add a backprop connection to this layer from the layer, _from.
       *
       * @param _from - the name of the layer to send its output
       */
      void
      add_backprop_connection(const std::shared_ptr<NetworkLayer>& _from, LayerConnRecord::ConnectionType _type = LayerConnRecord::Forward);

   protected:
      std::shared_ptr<BasicLayer> basic_layer;
      const std::valarray<double>& virtual_input_vector_const_ref = virtual_input_vector;
      const ValarrMap& input_map_const_ref = input_map;

      // Layer error accumulated from upstream layers and external NN error
      std::valarray<double> layer_errorv;

      // Local data member to hold scattered partial errors with
      // respect to the input to be back-propagated to layers that
      // feed activity to this layer.
      ValarrMap input_error_map;

   /* ***********************************************************
    * Private data members
    */
   private:
      bool output_layer_flag;

      std::vector<LayerConnRecord> activation_connections;
      std::vector<ExternalInputRecord> external_input_fields;
      std::vector<LayerConnRecord> backprop_connections;

      // Local valarray to contain the marshalled input vector for the layer
      std::valarray<double> virtual_input_vector;
      ValarrMap input_map;
   };

   inline std::shared_ptr<BasicLayer>& NetworkLayer::basiclayer()
   {
      return basic_layer;
   }

   inline
   void NetworkLayer::set_weight_initializer(std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)>& _func)
   {
      _weight_initializer = _func;
   }

   inline
   void NetworkLayer::set_biases(double _val)
   {
      basic_layer->layer_weights.set_biases(_val);
   }

   inline
   void NetworkLayer::set_biases(const std::valarray<double>& _biases)
   {
      basic_layer->layer_weights.set_biases(_biases);
   }

   inline
   void NetworkLayer::set_weights(double _val)
   {
      basic_layer->layer_weights.set(_val);
   }

   inline
   void NetworkLayer::set_weights(const Array2D<double>& _weights)
   {
      basic_layer->layer_weights.set(_weights);
   }

   inline void NetworkLayer::initialize_weights(void)
   {
      if (_weight_initializer)
      {
         Array2D<double>::Dimensions dims = basic_layer->layer_weights.size();
         basic_layer->layer_weights.set(_weight_initializer(dims.rows, dims.cols));
      }
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
   const ValarrMap& NetworkLayer::input_value_map() const
   {
      return input_map;
   }

   inline
   LayerWeights& NetworkLayer::weights()
   {
      return basic_layer->layer_weights;
   }

   inline
   const LayerWeights& NetworkLayer::weights() const
   {
      return basic_layer->layer_weights;
   }

   inline
   const LayerState& NetworkLayer::layer_state(void) const
   {
      return basic_layer->state();
   }

   inline
   const Array2D<double>& NetworkLayer::dy_dnet(void) const
   {
      return basic_layer->get_dy_dnet();
   }

   inline
   const Array2D<double>& NetworkLayer::dnet_dw(void) const
   {
      return basic_layer->get_dnet_dw();
   }

   inline
   const Array2D<double>& NetworkLayer::dnet_dx(void) const
   {
      return basic_layer->get_dnet_dx();
   }

   inline
   const Array2D<double>& NetworkLayer::dE_dw(void) const
   {
      return basic_layer->get_dE_dw();
   }

   inline
   void NetworkLayer::clear(void)
   {
      basic_layer.reset();
      input_map.clear();
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
