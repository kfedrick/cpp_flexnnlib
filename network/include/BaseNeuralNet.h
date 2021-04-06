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
      BaseNeuralNet(const BaseNeuralNet& _nnet);

      virtual ~BaseNeuralNet();

      void copy(const BaseNeuralNet& _nnet);

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
      virtual const void backprop(const ValarrMap& _egradient);

      const LayerWeights& get_weights(const std::string _layerid) const;

      void set_weights(const std::string _layerid, const LayerWeights& _weight);

      void adjust_weights(const std::string _layerid, const Array2D<double>& _deltaw);

      NetworkWeights get_weights(void) const;

      void set_weights(const NetworkWeights& _weight);

      const std::set<std::string>& get_layer_names(void);

      std::map<std::string, std::shared_ptr<NetworkLayer>>& get_layers(void);
      const std::map<std::string, std::shared_ptr<NetworkLayer>>& get_layers(void) const;

      std::string toJSON(void) const;

   protected:
      std::vector<std::shared_ptr<NetworkLayer>>& get_ordered_layers(void);

   private:
      size_t network_output_size;

      // Network topology
      NetworkTopology network_topology;

      // Set containing basic_layer names
      std::set<std::string> layer_name_set;

      // recurrent_network_flag - Set if this network has recurrent connections.
      bool recurrent_network_flag;

      // virtual_network_output_layer_ref - Used to gather network output from the output layers.
      NetworkOutput& virtual_network_output_layer_ref;
   };

   inline const std::set<std::string>& BaseNeuralNet::get_layer_names(void)
   {
      return layer_name_set;
   }

   inline
   std::map<std::string, std::shared_ptr<NetworkLayer>>& BaseNeuralNet::get_layers(void)
   {
      return network_topology.get_layers();
   }

   inline
   const std::map<std::string, std::shared_ptr<NetworkLayer>>& BaseNeuralNet::get_layers(void) const
   {
      return network_topology.get_layers();
   }

   inline
   std::vector<std::shared_ptr<NetworkLayer>>& BaseNeuralNet::get_ordered_layers(void)
   {
      return network_topology.get_ordered_layers();
   }

   inline size_t BaseNeuralNet::size(void)
   {
      return virtual_network_output_layer_ref.size();
   }

   inline const ValarrMap& BaseNeuralNet::value_map() const
   {
      return virtual_network_output_layer_ref.input_value_map();
   }
}

#endif //FLEX_NEURALNET_BASENEURALNET_H_
