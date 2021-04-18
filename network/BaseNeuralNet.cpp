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
   // TODO - implement
}

void
BaseNeuralNet::reset(void)
{
   // TODO - implement
}

const flexnnet::ValarrMap&
BaseNeuralNet::activate(const ValarrMap& _externin)
{
   /*
    * Activate all network layers
    */
   for (int i = 0; i < ordered_layers.size(); i++)
      const std::valarray<double>& temp = ordered_layers[i]->activate(_externin);

   return value_map();
}

const void
BaseNeuralNet::backprop(const ValarrMap& _egradient)
{
   /*
    * Backprop through all network layers in reverse activation order
    */
   for (int i = ordered_layers.size() - 1; i >= 0; i--)
      ordered_layers[i]->backprop(_egradient);
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