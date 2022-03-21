//
// Created by kfedrick on 4/10/21.
//

#ifndef FLEX_NEURALNET_NEURALNETTOPOLOGY_H_
#define FLEX_NEURALNET_NEURALNETTOPOLOGY_H_

#include <memory>
#include <string>
#include <map>
#include <vector>

#include <NetworkLayer.h>

namespace flexnnet
{
   class NeuralNetTopology
   {
   public:
      friend class NeuralNetBuilder;
      using NETWORK_LAYER_MAP_TYP = std::map<std::string, std::shared_ptr<NetworkLayer>>;

   public:
      NeuralNetTopology();
      NeuralNetTopology(const NeuralNetTopology& _topo);
      NeuralNetTopology& operator=(const NeuralNetTopology& _topo);
      virtual ~NeuralNetTopology();

      void clear(void);

      template <unsigned int N>
      std::array<std::string, N> get_output_layer_names() const;

   private:
      void copy(const NeuralNetTopology& _topo);
      void clone_layers(const NETWORK_LAYER_MAP_TYP& _layers);
      void copy_layer_connections(std::vector<LayerConnRecord>& _to, const std::vector<LayerConnRecord>& _from);

   public:
      // Network layers
      NETWORK_LAYER_MAP_TYP network_layers;

      // List of layers in order they will be activated during forward
      // network activation.
      std::vector<std::shared_ptr<NetworkLayer>> ordered_layers;

      // Ordered list of network output layers.
      std::vector<std::shared_ptr<NetworkLayer>> network_output_layers;
   };

   template<unsigned int N>
   std::array<std::string, N> NeuralNetTopology::get_output_layer_names() const
   {
      std::array<std::string, N> names;
      for (unsigned int ndx=0; ndx < network_output_layers.size(); ndx++)
         names[ndx] = network_output_layers[ndx]->name();

      return std::array<std::string, N>(names);
   }
}

#endif //FLEX_NEURALNET_NEURALNETTOPOLOGY_H_
