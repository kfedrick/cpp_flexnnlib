//
// Created by kfedrick on 2/20/21.
//
// NetworkTopology contains builder methods for specifying the
// layers and basic_layer connections for a base neural network and
// acts as a container for the network layers and basic_layer interconnection
// objects.
//

#ifndef FLEX_NEURALNET_NETWORKTOPOLOGY_H_
#define FLEX_NEURALNET_NETWORKTOPOLOGY_H_

#include <string>
#include <memory>
#include <iostream>
#include <sstream>
#include <map>
#include <set>

#include "BasicLayer.h"
#include "NetworkLayerImpl.h"
#include "NetworkOutput.h"

namespace flexnnet
{
   class NetworkTopology
   {
   public:
      void doit();

      /**
       * Create a topology for a neural network with the specified named external
       * input vectors, _xinput_sample.
       * @param _xinput
       */
      NetworkTopology(const NNetIO_Typ& _xinput_sample);
      NetworkTopology(const NetworkTopology& _topo);
      ~NetworkTopology();

      NetworkTopology& operator=(const NetworkTopology& _topo);

      /**
       * Clear all network topology components
       */
      void clear(void);

      /**
       * Add a new basic_layer to the network
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
       * Add a connection to the basic_layer, _to, from the basic_layer, _from.
       *
       * @param _to - the name of the basic_layer to recieve input
       * @param _from - the name of the basic_layer to send its output
       */
      void
      add_layer_connection(const std::string& _to, const std::string& _from, LayerConnRecord::ConnectionType _type = LayerConnRecord::Forward);

      /**
       * Add a connection to the basic_layer, _to, from an external input vector.
       * @param _to
       * @param _vec
       */
      void
      add_external_input_field(const std::string& _to, const std::string& _field);

      std::map<std::string, std::shared_ptr<NetworkLayerImpl>>& get_layers(void);

      const std::map<std::string, std::shared_ptr<NetworkLayerImpl>>& get_layers(void) const;

      NetworkOutput& get_network_output_layer(void);

      const NetworkOutput& get_network_output_layer(void) const;

      const std::vector<std::shared_ptr<NetworkLayerImpl>>& get_output_layers(void) const;

      std::vector<std::shared_ptr<NetworkLayerImpl>>& get_output_layers(void);

      /**
       * Get an ordered list of input basic_layer connections for the specified
       * basic_layer, _layer_id. This list specified basic_layer from which the specified
       * basic_layer receives inputs.
       * @param _layer_id
       * @return
       */
      const std::vector<LayerConnRecord>& get_activation_connections(const std::string& _layer_id) const;

      /**
       * Get an ordered list of backprop basic_layer connections for the specified
       * basic_layer, _layer_id. This list specifies the layers from which the
       * specified basic_layer receives backprop error.
       * @param _layer_id
       * @return
       */
      const std::vector<LayerConnRecord>& get_backprop_connections(const std::string& _layer_id) const;

      /**
       * Return an ordered list of field names from which the specified basic_layer, _layer_id,
       * receives external input.
       * @param _layer_id
       * @return
       */
      const std::vector<ExternalInputRecord>& get_external_input_fields(const std::string& _layer_id) const;

      /**
       * Return an list of layers in the correct activation order.
       * @return
       */
      const std::vector<std::shared_ptr<NetworkLayerImpl>>& get_ordered_layers(void) const;

      /**
       * Return an list of layers in the correct activation order.
       * @return
       */
      std::vector<std::shared_ptr<NetworkLayerImpl>>& get_ordered_layers(void);

   /*
    * Private helper functions
    */
   private:
      void copy(const NetworkTopology& _topo);
      std::map<std::string, std::shared_ptr<BasicLayer>> clone_baselayers(const NetworkTopology& _topo);
      std::shared_ptr<NetworkLayerImpl> copy_netlayer(const std::shared_ptr<NetworkLayerImpl>& _netlayer, const std::map<std::string, std::shared_ptr<BasicLayer>>& _baselayers);
      std::vector<LayerConnRecord> copy_connections(const std::vector<LayerConnRecord>& _conns, const std::map<std::string, std::shared_ptr<BasicLayer>>& _baselayers);

      void add_forward_connection(NetworkLayerImpl& _to, NetworkLayerImpl& _from, LayerConnRecord::ConnectionType _type, std::set<std::string>& _from_dependencies);
      void add_recurrent_connection(NetworkLayerImpl& _to, NetworkLayerImpl& _from, LayerConnRecord::ConnectionType _type, std::set<std::string>& _from_dependencies);
      void add_lateral_connection(NetworkLayerImpl& _to, NetworkLayerImpl& _from, LayerConnRecord::ConnectionType _type, std::set<std::string>& _to_dependencies, std::set<std::string>& _from_dependencies);

      /**
       * Return a set containing the names of layers directly through forward
       * connections, feeding activity into the basic_layer, _name.
       *
       * @param _dependencies
       * @param _name
       */
      void get_input_dependencies(std::set<std::string>& _dependencies, const std::string& _from);

      /**
       *
       */
      void update_activation_order(void);

      void update_network_output(void);

      /*
       * Private member data
       */
   private:
      // Network layers
      std::map<std::string, std::shared_ptr<NetworkLayerImpl>> network_layers;

      // network_output_conn - Used to coelesce network output from the output layers
      // and to scatter network backpropagation error to the output layers.
      //
      NetworkOutput network_output_layer;

      // List of layers in order they will be activated during forward
      // network activation.
      std::vector<std::shared_ptr<NetworkLayerImpl>> ordered_layers;

      // List of network output layers.
      std::vector<std::shared_ptr<NetworkLayerImpl>> network_output_layers;

      // Sample basic_layer input fields
      NNetIO_Typ sample_extern_input;
   };

   inline
   std::map<std::string, std::shared_ptr<NetworkLayerImpl>>& NetworkTopology::get_layers(void)
   {
      return network_layers;
   }

   inline
   const std::map<std::string, std::shared_ptr<NetworkLayerImpl>>& NetworkTopology::get_layers(void) const
   {
      return network_layers;
   }

   inline
   NetworkOutput& NetworkTopology::get_network_output_layer(void)
   {
      return network_output_layer;
   }

   inline
   const NetworkOutput& NetworkTopology::get_network_output_layer(void) const
   {
      return network_output_layer;
   }

   inline
   const std::vector<std::shared_ptr<NetworkLayerImpl>>& NetworkTopology::get_output_layers(void) const
   {
      return network_output_layers;
   }

   inline
   std::vector<std::shared_ptr<NetworkLayerImpl>>& NetworkTopology::get_output_layers(void)
   {
      return network_output_layers;
   }

   inline
   const std::vector<LayerConnRecord>&
   NetworkTopology::get_activation_connections(const std::string& _layer_id) const
   {
      return network_layers.at(_layer_id)->activation_connections;
   }

   inline
   const std::vector<LayerConnRecord>& NetworkTopology::get_backprop_connections(const std::string& _layer_id) const
   {
      return network_layers.at(_layer_id)->backprop_connections;
   }

   inline
   const std::vector<ExternalInputRecord>& NetworkTopology::get_external_input_fields(const std::string& _layer_id) const
   {
      return network_layers.at(_layer_id)->external_input_fields;
   }

   inline
   void NetworkTopology::clear(void)
   {
      network_layers.clear();
      network_output_layers.clear();
      ordered_layers.clear();
      sample_extern_input.clear();
   }

   template<class _LayerType> std::shared_ptr<_LayerType>
   NetworkTopology::add_layer(const std::string& _name, size_t _sz, bool _output, const typename _LayerType::Parameters& _params)
   {
      if (network_layers.find(_name) != network_layers.end())
      {
         static std::stringstream sout;
         sout << "Error : NetworkTopology::add_layer() - basic_layer \"" << _name.c_str() << "\" already exists."
              << std::endl;
         throw std::invalid_argument(sout.str());
      }

      auto layer_ptr = std::shared_ptr<_LayerType>(new _LayerType(_sz, _name));
      layer_ptr->set_params(_params);

      network_layers[_name] = std::shared_ptr<NetworkLayerImpl>(new NetworkLayerImpl(layer_ptr, _output));

      if (_output)
      {
         network_output_layers.push_back(network_layers[_name]);
         update_network_output();
      }

      return layer_ptr;
   }

   inline
   const std::vector<std::shared_ptr<NetworkLayerImpl>>& NetworkTopology::get_ordered_layers(void) const
   {
      return ordered_layers;
   }

   inline
   std::vector<std::shared_ptr<NetworkLayerImpl>>& NetworkTopology::get_ordered_layers(void)
   {
      return ordered_layers;
   }
}
#endif //FLEX_NEURALNET_NETWORKTOPOLOGY_H_
