//
// Created by kfedrick on 2/20/21.
//
// NetworkTopology contains builder methods for specifying the
// layers and basiclayer connections for a base neural network and
// acts as a container for the network layers and basiclayer interconnection
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
#include "NetworkLayer.h"

namespace flexnnet
{
   class NetworkTopology
   {
   public:

      /**
       * Create a topology for a neural network with the specified named external
       * input vectors, _xinput_sample.
       * @param _xinput
       */
      NetworkTopology(const std::map<std::string, std::vector<double>> _xinput_sample);
      ~NetworkTopology();

      void set(const NetworkTopology& _topo);
      NetworkTopology& operator=(const NetworkTopology& _topo);

      /**
       * Clear all network topology components
       */
      void clear(void);

      /**
       * Add a new basiclayer to the network
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
       * Add a connection to the basiclayer, _to, from the basiclayer, _from.
       *
       * @param _to - the name of the basiclayer to recieve input
       * @param _from - the name of the basiclayer to send its output
       */
      void
      add_layer_connection(const std::string& _to, const std::string& _from, LayerConnRecord::ConnectionType _type = LayerConnRecord::Forward);

      /**
       * Add a connection to the basiclayer, _to, from an external input vector.
       * @param _to
       * @param _vec
       */
      void
      add_external_input_field(const std::string& _to, const std::string& _field);

      const NetworkLayer& get_layer(const std::string& _id) const;

      NetworkLayer& get_layer(const std::string& _id);

      const std::vector<std::shared_ptr<NetworkLayer>>& get_output_layers(void) const;

      std::vector<std::shared_ptr<NetworkLayer>>& get_output_layers(void);

      /**
       * Get an ordered list of input basiclayer connections for the specified
       * basiclayer, _layer_id. This list specified basiclayer from which the specified
       * basiclayer receives inputs.
       * @param _layer_id
       * @return
       */
      const std::vector<LayerConnRecord>& get_activation_connections(const std::string& _layer_id) const;

      /**
       * Get an ordered list of backprop basiclayer connections for the specified
       * basiclayer, _layer_id. This list specifies the layers from which the
       * specified basiclayer receives backprop error.
       * @param _layer_id
       * @return
       */
      const std::vector<LayerConnRecord>& get_backprop_connections(const std::string& _layer_id) const;

      /**
       * Return an ordered list of field names from which the specified basiclayer, _layer_id,
       * receives external input.
       * @param _layer_id
       * @return
       */
      const std::vector<std::string>& get_external_input_fields(const std::string& _layer_id) const;

      /**
       * Return an list of layers in the correct activation order.
       * @return
       */
      const std::vector<std::shared_ptr<NetworkLayer>>& get_ordered_layers(void) const;

      /**
       * Return an list of layers in the correct activation order.
       * @return
       */
      std::vector<std::shared_ptr<NetworkLayer>>& get_ordered_layers(void);

   /*
    * Private helper functions
    */
   private:
      void add_forward_connection(NetworkLayer& _to, NetworkLayer& _from, LayerConnRecord::ConnectionType _type, std::set<std::string>& _from_dependencies);
      void add_recurrent_connection(NetworkLayer& _to, NetworkLayer& _from, LayerConnRecord::ConnectionType _type, std::set<std::string>& _from_dependencies);
      void add_lateral_connection(NetworkLayer& _to, NetworkLayer& _from, LayerConnRecord::ConnectionType _type, std::set<std::string>& _to_dependencies, std::set<std::string>& _from_dependencies);

      /**
       * Return a set containing the names of layers directly through forward
       * connections, feeding activity into the basiclayer, _name.
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
      std::map<std::string, std::shared_ptr<NetworkLayer>> layers;

      // List of network output layers.
      std::vector<std::shared_ptr<NetworkLayer>> network_output_layers;

      // List of layers in order they will be activated during forward
      // network activation.
      std::vector<std::shared_ptr<NetworkLayer>> ordered_layers;

      // Sample basiclayer input fields
      std::map<std::string, std::vector<double>> sample_extern_input;
   };

   inline
   const NetworkLayer& NetworkTopology::get_layer(const std::string& _id) const
   {
      return *layers.at(_id);
   }

   inline
   NetworkLayer& NetworkTopology::get_layer(const std::string& _id)
   {
      return *layers.at(_id);
   }

   inline
   const std::vector<std::shared_ptr<NetworkLayer>>& NetworkTopology::get_output_layers(void) const
   {
      return network_output_layers;
   }

   inline
   std::vector<std::shared_ptr<NetworkLayer>>& NetworkTopology::get_output_layers(void)
   {
      return network_output_layers;
   }

   inline
   const std::vector<LayerConnRecord>&
   NetworkTopology::get_activation_connections(const std::string& _layer_id) const
   {
      return layers.at(_layer_id)->get_activation_connections();
   }

   inline
   const std::vector<LayerConnRecord>& NetworkTopology::get_backprop_connections(const std::string& _layer_id) const
   {
      return layers.at(_layer_id)->get_backprop_connections();
   }

   inline
   const std::vector<std::string>& NetworkTopology::get_external_input_fields(const std::string& _layer_id) const
   {
      return layers.at(_layer_id)->get_external_input_fields();
   }

   inline
   void NetworkTopology::clear(void)
   {
      layers.clear();
      network_output_layers.clear();
      ordered_layers.clear();
      sample_extern_input.clear();
   }

   template<class _LayerType> std::shared_ptr<_LayerType>
   NetworkTopology::add_layer(const std::string& _name, size_t _sz, bool _output, const typename _LayerType::Parameters& _params)
   {
      if (layers.find(_name) != layers.end())
      {
         static std::stringstream sout;
         sout << "Error : NetworkTopology::add_layer() - basiclayer \"" << _name.c_str() << "\" already exists."
              << std::endl;
         throw std::invalid_argument(sout.str());
      }

      auto layer_ptr = std::shared_ptr<_LayerType>(new _LayerType(_sz, _name));
      layer_ptr->set_params(_params);

      layers[_name] = std::shared_ptr<NetworkLayer>(new NetworkLayer(layer_ptr, _output));

      if (_output)
         network_output_layers.push_back(layers[_name]);

      return layer_ptr;
   }

   inline
   const std::vector<std::shared_ptr<NetworkLayer>>& NetworkTopology::get_ordered_layers(void) const
   {
      return ordered_layers;
   }

   inline
   std::vector<std::shared_ptr<NetworkLayer>>& NetworkTopology::get_ordered_layers(void)
   {
      return ordered_layers;
   }
}
#endif //FLEX_NEURALNET_NETWORKTOPOLOGY_H_
