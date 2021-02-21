//
// Created by kfedrick on 2/20/21.
//

#ifndef FLEX_NEURALNET_BASENEURALNET_H_
#define FLEX_NEURALNET_BASENEURALNET_H_

#include "flexnnet.h"

namespace flexnnet
{
   typedef  std::map<std::string, std::valarray<double>> NNetIO_Typ;

   class BaseNeuralNet : public NamedObject
   {
   public:
      BaseNeuralNet(const std::string& _name = "BasicNeuralNet", const std::vector<std::shared_ptr<BasicLayer>>& _layers = {});
      virtual ~BaseNeuralNet();

   public:
      /**
       * BaseNeuralNet::size(void)
       *    Returns the size of the network output vector.
       *
       * @return - the size of the network output vector.
       */
      size_t size(void);

   public:

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
      virtual const NNetIO_Typ& activate(const NNetIO_Typ& _input);

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

   public:
      std::string toJSON(void) const;
      const std::vector<std::shared_ptr<BasicLayer>> get_layers(void) const;

   private:
      /**
       * Private helper function to initialize network_output_conn by adding connections
       * from to network_output_conn from all output layers.
       *
       * @return - the size of the virtual output vector.
       */
      std::map<std::string, std::valarray<double>> init_network_output_layer(void);

   private:
      size_t network_output_size;

      // Network layers stored in proper activation order
      std::vector<std::shared_ptr<BasicLayer>> network_layers;

      // Set containing layer names
      std::set<std::string> layer_name_set;

      // recurrent_network_flag - Set if this network has recurrent connections.
      bool recurrent_network_flag;

      // network_output_conn - Used to coelesce network output from the output layers
      // and to scatter network backpropagation error to the output layers.
      //
      NetworkOutput network_output_conn;

      NNetIO_Typ network_output;
   };

   inline const std::set<std::string>& BaseNeuralNet::get_layer_names(void)
   {
      return layer_name_set;
   }

   inline size_t BaseNeuralNet::size(void)
   {
      return network_output_conn.virtual_input_size();
   }

   inline const std::vector<std::shared_ptr<BasicLayer>> BaseNeuralNet::get_layers(void) const
   {
      return network_layers;
   }
}

#endif //FLEX_NEURALNET_BASENEURALNET_H_
