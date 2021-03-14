//
// Created by kfedrick on 2/20/21.
//

#ifndef FLEX_NEURALNET_BASENEURALNET_H_
#define FLEX_NEURALNET_BASENEURALNET_H_

#include "flexnnet.h"
#include "NetworkTopology.h"
#include "NetworkOutput.h"
#include <ValarrayMap.h>

namespace flexnnet
{
   class BaseNeuralNet
   {
   public:
      BaseNeuralNet(const NetworkTopology& _topology = NetworkTopology({}));
      virtual ~BaseNeuralNet();

      /**
       * BaseNeuralNet::size(void)
       *    Returns the size of the network output vector.
       *
       * @return - the size of the network output vector.
       */
      size_t size(void);

      virtual const ValarrMap& value_map() const;

      /**
       * Initialize network weights and other network state information
       * prior to training.
       */
      virtual void initialize_weights(void);

      /**
       * Reset network state activation prior to presentation of next
       * exemplar or episode.
       */
      virtual void reset(void);

      /**
       * Activate the network with the specified input data
       *
       * @param _indatum - the input datum to the network
       * @return - the network output for the specified input data
       */
      virtual const ValarrMap& activate(const ValarrMap& _input);

      /**
       * Calculate the Jacobian for most recent network activation as the partial
       * derivatives of the specified gradient with respect to the weights. It
       * an error gradient is specified the Jacobian is the set of partial derivatives
       * of the weights with respect to the output error; if a unity vector is provided
       * the Jacobian is the set of partial derivatives of the weights with respect
       * to the network output.
       *
       * @param _gradient
       */
      virtual const void backprop(const std::valarray<double>& _gradient);

      NetworkWeights get_weights(void) const;
      void set_weights(const NetworkWeights& _weight);

      const std::set<std::string>& get_layer_names(void);

      std::map<std::string, std::shared_ptr<NetworkLayer>>& get_layers(void);

      std::string toJSON(void) const;

   protected:
      std::vector<std::shared_ptr<NetworkLayer>>& get_ordered_layers(void);

   private:
      /**
       * Private helper function to initialize network_output_conn by adding connections
       * from to network_output_conn from all output layers.
       *
       * @return - the size of the virtual output vector.
       */
      void init_network_output_layer(void);

   private:
      size_t network_output_size;

      // Network topology
      NetworkTopology network_topology;

      // Set containing basic_layer names
      std::set<std::string> layer_name_set;

      // recurrent_network_flag - Set if this network has recurrent connections.
      bool recurrent_network_flag;

      // network_output_layer - Used to gather network output from the output layers.
      NetworkOutput& network_output_layer = network_topology.get_network_output_layer();
   };

   inline const std::set<std::string>& BaseNeuralNet::get_layer_names(void)
   {
      return layer_name_set;
   }

   inline
   std::map<std::string, std::shared_ptr<NetworkLayer>>& BaseNeuralNet::get_layers(void)
   {
      return reinterpret_cast<std::map<std::string, std::shared_ptr<NetworkLayer>>&>(network_topology.get_layers());
   }

   inline
   std::vector<std::shared_ptr<NetworkLayer>>& BaseNeuralNet::get_ordered_layers(void)
   {
      return reinterpret_cast<std::vector<std::shared_ptr<NetworkLayer>>&>(network_topology.get_ordered_layers());
   }

   inline size_t BaseNeuralNet::size(void)
   {
      return network_output_layer.size();
   }

   inline const ValarrMap& BaseNeuralNet::value_map() const
   {
      return network_output_layer.input_value_map();
   }
}

#endif //FLEX_NEURALNET_BASENEURALNET_H_
