//
// Created by kfedrick on 2/20/21.
//
// NetworkTopology contains builder methods for specifying the
// layers and layer connections for a base neural network and
// acts as a container for the network layers and layer interconnection
// objects.
//

#ifndef FLEX_NEURALNET_NETWORKTOPOLOGY_H_
#define FLEX_NEURALNET_NETWORKTOPOLOGY_H_

#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <set>

#include "BasicLayer.h"
#include "BaseNeuralNet.h"

namespace flexnnet
{
   class NetworkTopology
   {
   public:
      enum ConnectionType
      {
         Forward = 0, Recurrent = 1, Lateral = 2
      };

      class LayerConnRecord
      {
      public:
         LayerConnRecord(std::shared_ptr<BasicLayer>& _layer, ConnectionType _type)
         {
            layer = _layer;
            connection_type = _type;
         }

         bool is_recurrent() const
         {
            return !(connection_type == Forward);
         }

         ConnectionType connection_type;
         std::shared_ptr<BasicLayer> layer;
      };

   public:

      /**
       * Create a topology for a neural network with the specified named external
       * input vectors, _xinput_sample.
       * @param _xinput
       */
      NetworkTopology(const std::map<std::string, std::vector<double>> _xinput_sample);
      ~NetworkTopology();

      /**
       * Clear all network topology components
       */
      void clear(void);

      /**
       * Add a new layer to the network
       *
       * @tparam _LayerType
       * @param _sz
       * @param _name
       * @param _params
       * @return
       */
      template<class _LayerType> std::shared_ptr<_LayerType>
      add_layer(const std::string& _name, size_t _sz, bool _output = false, const typename _LayerType::Parameters& _params = _LayerType::DEFAULT_PARAMS);

      /**
       * Add a connection to the layer, _to, from the layer, _from.
       *
       * @param _to - the name of the layer to recieve input
       * @param _from - the name of the layer to send its output
       */
      void
      add_layer_connection(const std::string& _to, const std::string& _from, ConnectionType _type = Forward);

      /**
       * Add a connection to the layer, _to, from an external input vector.
       * @param _to
       * @param _vec
       */
      void
      add_external_input_field(const std::string& _to, const std::string& _field);

      const std::map<std::string, std::shared_ptr<BasicLayer>>& get_layers() const;

      const std::vector<std::shared_ptr<BasicLayer>>& get_output_layers(void) const;

      std::vector<std::shared_ptr<BasicLayer>>& get_output_layers(void);

      /**
       * Get an ordered list of input layer connections for the specified
       * layer, _layer_id. This list specified layer from which the specified
       * layer receives inputs.
       * @param _layer_id
       * @return
       */
      const std::vector<LayerConnRecord>& get_activation_connections(const std::string& _layer_id) const;

      /**
       * Get an ordered list of backprop layer connections for the specified
       * layer, _layer_id. This list specifies the layers from which the
       * specified layer receives backprop error.
       * @param _layer_id
       * @return
       */
      const std::vector<LayerConnRecord>& get_backprop_connections(const std::string& _layer_id) const;

      /**
       * Return an ordered list of field names from which the specified layer, _layer_id,
       * receives external input.
       * @param _layer_id
       * @return
       */
      const std::vector<std::string>& get_external_input_fields(const std::string& _layer_id) const;

      /**
       * Return an list of layers in the correct activation order.
       * @return
       */
      const std::vector<std::shared_ptr<BasicLayer>>& get_ordered_layers(void) const;

      /**
       * Return an list of layers in the correct activation order.
       * @return
       */
      std::vector<std::shared_ptr<BasicLayer>>& get_ordered_layers(void);

   /*
    * Private helper functions
    */
   private:
      void add_forward_connection(const std::string& _to, const std::string& _from, ConnectionType _type, std::set<std::string>& _from_dependencies);
      void add_recurrent_connection(const std::string& _to, const std::string& _from, ConnectionType _type, std::set<std::string>& _from_dependencies);
      void add_lateral_connection(const std::string& _to, const std::string& _from, ConnectionType _type, std::set<std::string>& _to_dependencies, std::set<std::string>& _from_dependencies);

      void insert_activation_connection(const std::string& _to, const std::string& _from, ConnectionType _type);
      void insert_backprop_connection(const std::string& _to, const std::string& _from, ConnectionType _type);

      /**
       * Return a set containing the names of layers directly through forward
       * connections, feeding activity into the layer, _name.
       *
       * @param _dependencies
       * @param _name
       */
      void getInputDependencies(std::set<std::string>& _dependencies, const std::string& _from);

      /**
       *
       */
      void update_activation_order(void);

      /*
       * Private member data
       */
   private:

      // Network layers
      std::map<std::string, std::shared_ptr<BasicLayer>> layers;

      // List of network output layers.
      std::vector<std::shared_ptr<BasicLayer>> network_output_layers;

      // List of layers in order they will be activated during forward
      // network activation.
      std::vector<std::shared_ptr<BasicLayer>> ordered_layers;

      // Map each network layer to the ordered list of layers
      // from which it receives an input connection.
      std::map<std::string, std::vector<LayerConnRecord>> activation_conn_map;

      // Map each network layer to the ordered list of layers that
      // receive input connections from this layer.
      std::map<std::string, std::vector<LayerConnRecord>> backprop_conn_map;

      // Map each network layer to the ordered list of external
      // input fields from which it receives input.
      std::map<std::string, std::vector<std::string>> external_input_fields_conn_map;

      // Sample layer input fields
      std::map<std::string, std::vector<double>> sample_extern_input;
   };

   inline
   const std::map<std::string, std::shared_ptr<BasicLayer>>& NetworkTopology::get_layers() const
   {
      return layers;
   }

   inline
   const std::vector<std::shared_ptr<BasicLayer>>& NetworkTopology::get_output_layers(void) const
   {
      return network_output_layers;
   }

   inline
   std::vector<std::shared_ptr<BasicLayer>>& NetworkTopology::get_output_layers(void)
   {
      return network_output_layers;
   }

   inline
   const std::vector<flexnnet::NetworkTopology::LayerConnRecord>&
   NetworkTopology::get_activation_connections(const std::string& _layer_id) const
   {
      return activation_conn_map.at(_layer_id);
   }

   inline
   const std::vector<flexnnet::NetworkTopology::LayerConnRecord>& NetworkTopology::get_backprop_connections(const std::string& _layer_id) const
   {
      return backprop_conn_map.at(_layer_id);
   }

   inline
   const std::vector<std::string>& NetworkTopology::get_external_input_fields(const std::string& _layer_id) const
   {
      return external_input_fields_conn_map.at(_layer_id);
   }

   inline
   void NetworkTopology::clear(void)
   {
      layers.clear();
      network_output_layers.clear();
      ordered_layers.clear();
      activation_conn_map.clear();
      backprop_conn_map.clear();
      sample_extern_input.clear();
   }

   template<class _LayerType> std::shared_ptr<_LayerType>
   NetworkTopology::add_layer(const std::string& _name, size_t _sz, bool _output, const typename _LayerType::Parameters& _params)
   {
      if (layers.find(_name) != layers.end())
      {
         static std::stringstream sout;
         sout << "Error : NetworkTopology::add_layer() - layer \"" << _name.c_str() << "\" already exists."
              << std::endl;
         throw std::invalid_argument(sout.str());
      }

      auto layer_ptr = std::shared_ptr<_LayerType>(new _LayerType(_sz, _name));
      layer_ptr->set_params(_params);

      layers[_name] = layer_ptr;

      if (_output)
         network_output_layers.push_back(layer_ptr);

      return layer_ptr;
   }

   inline
   const std::vector<std::shared_ptr<BasicLayer>>& NetworkTopology::get_ordered_layers(void) const
   {
      return ordered_layers;
   }

   inline
   std::vector<std::shared_ptr<BasicLayer>>& NetworkTopology::get_ordered_layers(void)
   {
      return ordered_layers;
   }
}
#endif //FLEX_NEURALNET_NETWORKTOPOLOGY_H_
