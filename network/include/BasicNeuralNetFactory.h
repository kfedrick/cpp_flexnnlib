//
// Created by kfedrick on 5/28/19.
//

#ifndef FLEX_NEURALNET_BASICNEURALNETFACTORY_H_
#define FLEX_NEURALNET_BASICNEURALNETFACTORY_H_

#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <set>

#include "BasicLayer.h"
#include "BasicNeuralNet.h"
#include "NetworkLayer.h"

namespace flexnnet
{
   class BasicNeuralNetFactory
   {
   public:
      BasicNeuralNetFactory();

   public:
      ~BasicNeuralNetFactory();

      void clear(void);

      template<class _LayerType> std::shared_ptr<_LayerType>
      add_layer(size_t _sz, const std::string& _name, const typename _LayerType::Parameters& _params = _LayerType::DEFAULT_PARAMS);

      /**
       * Add connection to the layer, _to, from the layer, _from.
       *
       * @param _to - the name of the layer to recieve input
       * @param _from - the name of the layer to send its output
       */
      void
      add_layer_connection(const std::string& _to, const std::string& _from, LayerConnRecord::ConnectionType _type = LayerConnRecord::Forward);

      /**
       * Add a connection to the layer, _to, from an external input vector.
       * @param _to
       * @param _vec
       */
      void
      set_layer_external_input(const std::string& _to, const Datum& _network_input, const std::set<std::string>& _patternNdx);

      /**
       * Build and return a pointer to the network
       * @return
       */
      std::shared_ptr<BasicNeuralNet> build(const std::string& _network_name);

      int size(void);

      std::vector<std::string> getActivationOrder(void);

   private:
      void updateActivationOrder(void);

      void getAllDependencies(std::set<std::string>& _dependencies, const std::string& _name);
      void getForwardDependencies(std::set<std::string>& _dependencies, const std::string& _name);
      void validate(void);
      void validateNetworkInput(const Datum& _network_input);
      unsigned int outputLayerCount();
      std::set<std::string> checkLayerInputSize();

      /**
       * @param _network_input
       */
      void set_network_input(const Datum& _network_input);


      /**
       * Private build artifacts
       */
   private:

      const size_t PLACEHOLDER_INPUT_SZ = 1;

      BasicNeuralNet* net;

      Datum network_input;

      std::map<std::string, std::shared_ptr<NetworkLayer>> layers;
      std::vector<std::string> layer_activation_order;

      bool network_input_set;
      bool layer_external_input_set;
      bool recurrent_network_flag;
      bool built;
   };

   inline int BasicNeuralNetFactory::size(void)
   {
      return layers.size();
   }

   template<class _TransFunc> std::shared_ptr<_TransFunc>
   BasicNeuralNetFactory::add_layer(size_t _sz, const std::string& _name, const typename _TransFunc::Parameters& _params)
   {
      if (layers.find(_name) != layers.end())
      {
         static std::stringstream sout;
         sout << "Error : BasicNeuralNetFactory::add_layer() - layer \"" << _name.c_str() << "\" already exists."
              << std::endl;
         throw std::invalid_argument(sout.str());
      }

      auto layer_ptr = std::shared_ptr<_TransFunc>(new _TransFunc(_sz, _name));
      layer_ptr->set_params(_params);

      // Add the new layer to the collection of network layers
//      layers[_name] = NetworkLayer(layer_ptr);
      layers[_name] = layer_ptr;

      return layer_ptr;
   }

   inline std::vector<std::string> BasicNeuralNetFactory::getActivationOrder(void)
   {
      return layer_activation_order;
   }
}

#endif //FLEX_NEURALNET_BASICNEURALNETFACTORY_H_
