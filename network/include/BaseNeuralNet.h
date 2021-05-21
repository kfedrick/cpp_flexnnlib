//
// Created by kfedrick on 4/11/21.
//

#ifndef FLEX_NEURALNET_BASENEURALNET_H_
#define FLEX_NEURALNET_BASENEURALNET_H_

#include <flexnnet.h>
#include <NeuralNetTopology.h>

namespace flexnnet
{
   class BaseNeuralNet : protected NeuralNetTopology
   {
   public:
      BaseNeuralNet();
      BaseNeuralNet(const NeuralNetTopology& _topology);
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

      // TODO - Do this next - it's iterable and indexable if you know the order. Use index for input in base NN
      //virtual const std::vector<std::pair<std::string,std::valarray<double>>>& activate2(const std::vector<std::valarray<double>>& _input);


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

      /**
       * Initialize network weights and other network state information
       * prior to training.
       */
      virtual void initialize_weights(void);

      void set_weights(const std::string _layerid, const LayerWeights& _weight);

      void adjust_weights(const std::string _layerid, const Array2D<double>& _deltaw);

      const std::set<std::string>& get_layer_names(void);

      const std::map<std::string, std::shared_ptr<NetworkLayer>>& get_layers(void) const;

   protected:
      std::vector<std::shared_ptr<NetworkLayer>>& get_ordered_layers(void);

   private:
      size_t network_output_size;

      // Set containing basic_layer names
      std::set<std::string> layer_name_set;

      // recurrent_network_flag - Set if this network has recurrent connections.
      bool recurrent_network_flag;

      mutable ValarrMap output_val_map;
   };

   inline
   const std::map<std::string, std::shared_ptr<NetworkLayer>>& BaseNeuralNet::get_layers(void) const
   {
      return network_layers;
   }

   inline const std::set<std::string>& BaseNeuralNet::get_layer_names(void)
   {
      return layer_name_set;
   }

   inline size_t BaseNeuralNet::size(void)
   {
      // TODO - fix ineffecient implementation
      size_t sz = 0;
      for (auto& it : network_output_layers)
         sz += it->value().size();

      return sz;
   }

   inline const ValarrMap& BaseNeuralNet::value_map() const
   {
      // TODO - fix ineffecient implementation
      for (auto& it : network_output_layers)
         output_val_map[it->name()] = it->value();

      return output_val_map;
   }
}

#endif //FLEX_NEURALNET_BASENEURALNET_H_
