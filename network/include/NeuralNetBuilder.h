//
// Created by kfedrick on 4/10/21.
//

#ifndef FLEX_NEURALNET_NEURALNETBUILDER_H_
#define FLEX_NEURALNET_NEURALNETBUILDER_H_

#include <memory>

#include <flexnnet.h>
#include <NetworkLayer.h>
#include <NetworkLayerImpl.h>
#include <NeuralNet.h>
#include "NeuralNetTopology.h"

namespace flexnnet
{
   class NeuralNetBuilder
   {
   public:
      NeuralNetBuilder();
      NeuralNetBuilder(const ValarrMap& _xinput_sample);
      virtual ~NeuralNetBuilder(void);

      /**
       * Clear all network components
       */
      void clear(void);

      /**
       * Add a new layer to the network
       *
       * @tparam _LayerType
       * @param _name - layer id.
       * @param _sz - layer size (number of neurons in layer)
       * @param _output - output layer flag.
       * @param _params - layer parameters
       * @return
       */
      template<class _LayerType>
      std::shared_ptr<NetworkLayer>
      add_layer(const std::string& _name, size_t _sz, bool _output = false, const typename _LayerType::Parameters& _params = _LayerType::DEFAULT_PARAMS);

      /**
       * Add an input connection to the network layer, _to, from the layer, _from.
       *
       * @param _to - the name of the network layer to receive input
       * @param _from - the name of the network layer to send its output
       */
      void
      add_layer_connection(const std::string& _to, const std::string& _from, LayerConnRecord::ConnectionType _type = LayerConnRecord::Forward);

      /**
       * Add a connection from an external input vector to the network layer, _to.
       * @param _to - network layer id.
       * @param _field - external input field id.
       */
      void
      add_external_input_connection(const std::string& _to, const std::string& _field);

      template<typename _InTyp, typename _OutTyp>
      NeuralNet<_InTyp, _OutTyp> build(void);

   private:
      void validate_forward_connection(const std::string& _to, const std::string& _from, const std::set<std::string>& _from_dep);
      void validate_recurrent_connection(const std::string& _to, const std::string& _from, const std::set<std::string>& _from_dep);
      void validate_lateral_connection(const std::string& _to, const std::string& _from, std::set<std::string>& _to_dependencies, std::set<std::string>& _from_dependencies);

      /**
       * Update the correct activation order of the network layers
       */
      void update_activation_order(void);

      /**
       * Return a set containing the names of layers directly through forward
       * connections, feeding activity into the basic_layer, _name.
       *
       * @param _dependencies
       * @param _name
       */
      void get_input_dependencies(std::set<std::string>& _dependencies, const std::string& _from);

      void alloc_layer_data(void);

      unsigned int calc_input_size(const std::string& _id) const;

   private:
      ValarrMap sample_external_input;

      // Working copy of network topology
      NeuralNetTopology network_topology;

      // Convenience references to data members of working topology
      NeuralNetTopology::NETWORK_LAYER_MAP_TYP& topo_layers = network_topology.network_layers;
      std::vector<std::shared_ptr<NetworkLayer>>& topo_output_layers = network_topology.network_output_layers;
      std::vector<std::shared_ptr<NetworkLayer>>& topo_ordered_layers = network_topology.ordered_layers;
   };

   inline
   void NeuralNetBuilder::clear(void)
   {
      network_topology.clear();
   }

   template<class _LayerType>
   inline
   std::shared_ptr<NetworkLayer>
   NeuralNetBuilder::add_layer(const std::string& _name, size_t _sz, bool _output, const typename _LayerType::Parameters& _params)
   {
      NeuralNetTopology::NETWORK_LAYER_MAP_TYP& network_layers = network_topology.network_layers;
      std::vector<std::shared_ptr<NetworkLayer>>& nn_output_layers = network_topology.network_output_layers;

      if (topo_layers.find(_name) != topo_layers.end())
      {
         static std::stringstream sout;
         sout << "Error : NeuralNetBuilder::add_layer() - layer \"" << _name << "\" already exists."
              << std::endl;
         throw std::invalid_argument(sout.str());
      }

      auto layer_ptr = std::make_shared<NetworkLayerImpl<_LayerType>>(NetworkLayerImpl<_LayerType>(_sz, _name, _params, _output));
      topo_layers[_name] = layer_ptr;

      if (_output)
         topo_output_layers.push_back(network_layers[_name]);

      return layer_ptr;
   }

   template<typename _InTyp, typename _OutTyp>
   NeuralNet<_InTyp, _OutTyp> NeuralNetBuilder::build(void)
   {
      alloc_layer_data();
      return NeuralNet<_InTyp, _OutTyp>(network_topology);
   }
}

#endif //FLEX_NEURALNET_NEURALNETBUILDER_H_
