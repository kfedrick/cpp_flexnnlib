//
// Created by kfedrick on 4/11/21.
//

#include "BaseNeuralNet.h"
#include <NRandArray2DInitializer.h>
#include <Globals.h>

using flexnnet::BaseNeuralNet;

BaseNeuralNet::BaseNeuralNet() : NeuralNetTopology()
{
}

BaseNeuralNet::BaseNeuralNet(const flexnnet::NeuralNetTopology& _topology)
   : NeuralNetTopology(_topology)
{
   // Initialize dEdx based on layer info
   for (auto& layer_ptr : network_layers)
   {
      const ValarrMap& layer_dEdx = layer_ptr.second->dEdx();
      for (auto dEdx_entry : layer_dEdx)
         dEdx[dEdx_entry.first] = dEdx_entry.second;
   }
}

BaseNeuralNet::BaseNeuralNet(const BaseNeuralNet& _nnet)
   : NeuralNetTopology(_nnet)
{
   copy(_nnet);
}

BaseNeuralNet::~BaseNeuralNet()
{
}

void
BaseNeuralNet::copy(const BaseNeuralNet& _nnet)
{
   // Initialize dEdx based on layer info
   for (auto& layer_ptr : network_layers)
   {
      const ValarrMap& layer_dEdx = layer_ptr.second->dEdx();
      for (auto dEdx_entry : layer_dEdx)
         dEdx[dEdx_entry.first] = dEdx_entry.second;
   }
}

void
BaseNeuralNet::reset(void)
{
   // TODO - implement
}

const flexnnet::ValarrMap&
BaseNeuralNet::activate(const ValarrMap& _externin)
{

/*   std::cout << "NN::externin:\n";
   for (auto entry : _externin)
   {
      std::cout << "   " << entry.first << ": ";
      for (int i = 0; i < entry.second.size(); i++)
         std::cout << entry.second[i] << ", ";
      std::cout << "\n";
   }
   std::cout << "\n";*/

   //std::cout << "BaseNeuralNet.activate()\n" << std::flush;

   /*
    * Activate all network layers
    */
   for (int i = 0; i < ordered_layers.size(); i++)
      const std::valarray<double>& temp = ordered_layers[i]->activate(_externin);

/*   std::cout << "network out:\n";
   const ValarrMap& nnout = value_map();
   for (auto& v : nnout)
      for (auto& val : v.second)
         std::cout << "  " << val << ", ";
   std::cout << "--\n";*/

   //std::cout << "BaseNeuralNet.activate() EXIT\n" << std::flush;

   return value_map();
}

const void
BaseNeuralNet::backprop(const ValarrMap& _egradient)
{
   // Clear error partial derivatives for network inputs
   for (auto entry : dEdx)
      dEdx[entry.first] = 0;

   /*
    * Backprop through all network layers in reverse activation order
    */
   for (int i = ordered_layers.size() - 1; i >= 0; i--)
   {
      std::shared_ptr<NetworkLayer> layer = ordered_layers[i];
      auto& id = layer->name();
      layer->backprop(_egradient);

      // Accumulate error derivatives for network inputs
      const ValarrMap& dEdx_map = layer->dEdx();
      for (auto entry : dEdx_map)
         dEdx[entry.first] += dEdx_map.at(entry.first);
   }
}

const flexnnet::LayerWeights&
BaseNeuralNet::get_weights(const std::string _layerid) const
{
   return network_layers.at(_layerid)->weights();
}

void
BaseNeuralNet::initialize_weights(void)
{
   std::function<Array2D<double>(unsigned int, unsigned int)> f2 = random_2darray<double>;

   for (auto& layer_ptr : network_layers)
   {
      layer_ptr.second->set_weight_initializer(f2);
      layer_ptr.second->initialize_weights();
   }
}

void
BaseNeuralNet::set_weights(const std::string _layerid, const LayerWeights& _weights)
{
   // TODO - Validate layer id
   network_layers.at(_layerid)->set_weights(_weights.const_weights_ref);
}

void
BaseNeuralNet::adjust_weights(const std::string _layerid, const Array2D<double>& _deltaw)
{
   // TODO - Validate layer id
   network_layers.at(_layerid)->adjust_weights(_deltaw);
}