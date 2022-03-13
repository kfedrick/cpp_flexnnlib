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
}

#endif //FLEX_NEURALNET_NEURALNETTOPOLOGY_H_
