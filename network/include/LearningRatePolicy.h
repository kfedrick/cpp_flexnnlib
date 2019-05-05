/*
 * NetworkLayerLearningRate.h
 *
 *  Created on: May 3, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_LRATE_POLICY_H_
#define FLEX_NEURALNET_LRATE_POLICY_H_

#include "Array.h"
#include "BaseNeuralNet.h"

#include <cstddef>
#include <iostream>
#include <vector>
#include <map>

using namespace std;

namespace flex_neuralnet
{

/*
 * Learning rate policy for a layer. Responsible for providing the
 * learning rates for the biases and layer weights for a layer, and
 * for updating the learning rates according to policy. The base class
 * implements a fixed,non-adaptive learning rate policy.
 */

class LearningRatePolicy
{
public:
   LearningRatePolicy();
   LearningRatePolicy(const BaseNeuralNet& _nn);
   LearningRatePolicy(const LearningRatePolicy& _nnLRPolicy);
   virtual ~LearningRatePolicy();

   const map< string, vector<double> >& get_bias_learning_rates() const;
   const map< string, Array<double> >& get_weight_learning_rates() const;

   virtual void set_global_learning_rate(double _rate);
   virtual void set_layer_learning_rate(const string& _layerID, double _rate);
   virtual void set_layer_biases_learning_rate(const string& _layerID, double _rate);
   virtual void set_layer_weights_learning_rate(const string& _layerID, double _rate);

   LearningRatePolicy& operator=(const LearningRatePolicy& _nnLRPolicy);

   virtual void reset();
   virtual void update_learning_rate_adjustments(unsigned int _timeStep = 1);
   virtual void apply_learning_rate_adjustments();
   virtual void reduce_learning_rate(double _reductionFactor);

protected:
   void copy(const LearningRatePolicy& _nnLR);

private:
   void alloc_storage();

protected:
   const BaseNeuralNet* neural_net;
   map<string, vector<double> > layer_bias_learning_rates_map;
   map<string, Array<double> > layer_weight_learning_rates_map;
};

inline LearningRatePolicy::LearningRatePolicy()
{
   neural_net = NULL;
}

inline LearningRatePolicy::LearningRatePolicy(
      const BaseNeuralNet& _nn)
{
   neural_net = &_nn;
   alloc_storage();
}

inline
LearningRatePolicy::LearningRatePolicy(
      const LearningRatePolicy& _nnLRPolicy)
{
   copy(_nnLRPolicy);
}

inline LearningRatePolicy::~LearningRatePolicy()
{
   /*
    * Set pointer to layer to null to make sure nothing tries to
    * deallocate it.
    */
   neural_net = NULL;
}

inline
void LearningRatePolicy::alloc_storage()
{
   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();

   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      layer_bias_learning_rates_map[name].resize(layer.size(), 0.0);

      const Array<double>& layer_weights = layer.get_weights();
      layer_weight_learning_rates_map[name].resize(layer_weights.rowDim(),
            layer_weights.colDim(), 0.0);
   }
}

inline
void LearningRatePolicy::copy(
      const LearningRatePolicy& _nnLRPolicy)
{
   neural_net = _nnLRPolicy.neural_net;

   layer_bias_learning_rates_map = _nnLRPolicy.layer_bias_learning_rates_map;
   layer_weight_learning_rates_map = _nnLRPolicy.layer_weight_learning_rates_map;
}

inline
void LearningRatePolicy::set_global_learning_rate(double _rate)
{
   if (_rate < 0.0)
      throw invalid_argument("Error : learning rate cannot be negative");

   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      layer_bias_learning_rates_map[name].assign(layer_bias_learning_rates_map[name].size(),
            _rate);
      layer_weight_learning_rates_map[name] = _rate;
   }
}

inline
void LearningRatePolicy::set_layer_learning_rate(const string& _layerID, double _rate)
{
   set_layer_biases_learning_rate(_layerID, _rate);
   set_layer_weights_learning_rate(_layerID, _rate);
}

inline
void LearningRatePolicy::set_layer_biases_learning_rate(const string& _layerID, double _rate)
{
   if (layer_bias_learning_rates_map.find(_layerID) == layer_bias_learning_rates_map.end())
   {
      ostringstream err_str;
      err_str << "Error (set_layer_biases_learning_rate() : Layer ID (" << _layerID.c_str() << ") not found.";
      throw invalid_argument(err_str.str());
   }

   vector<double>& bias_rates = layer_bias_learning_rates_map[_layerID];
   bias_rates.assign(bias_rates.size(), _rate);
}

inline
void LearningRatePolicy::set_layer_weights_learning_rate(const string& _layerID, double _rate)
{
   if (layer_weight_learning_rates_map.find(_layerID) == layer_weight_learning_rates_map.end())
   {
      ostringstream err_str;
      err_str << "Error (set_layer_weights_learning_rate() : Layer ID (" << _layerID.c_str() << ") not found.";
      throw invalid_argument(err_str.str());
   }

   layer_weight_learning_rates_map[_layerID] = _rate;
}

inline const map<string, vector<double> >& LearningRatePolicy::get_bias_learning_rates() const
{
   return layer_bias_learning_rates_map;
}

inline const map<string, Array<double> >& LearningRatePolicy::get_weight_learning_rates() const
{
   return layer_weight_learning_rates_map;
}

inline LearningRatePolicy& LearningRatePolicy::operator=(
      const LearningRatePolicy& _nnLRPolicy)
{
   copy(_nnLRPolicy);
   return *this;
}

inline
void LearningRatePolicy::reset()
{
   // Base class implements a fixed, non-adaptive policy. Do nothing.
}

inline
void LearningRatePolicy::update_learning_rate_adjustments(
      unsigned int _timeStep)
{
   // Base class implements a fixed, non-adaptive policy. Do nothing.
}

inline
void LearningRatePolicy::apply_learning_rate_adjustments()
{
   // Base class implements a fixed, non-adaptive policy. Do nothing.
}

inline
void LearningRatePolicy::reduce_learning_rate(double _reductionFactor)
{
   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      vector<double>& layer_bias_learning_rates = layer_bias_learning_rates_map[name];
      for (unsigned int ndx = 0; ndx < layer_bias_learning_rates.size(); ndx++)
      {
         layer_bias_learning_rates[ndx] *= _reductionFactor;
      }

      Array<double>& layer_weight_learning_rates =
            layer_weight_learning_rates_map[name];
      for (unsigned int to = 0; to < layer_weight_learning_rates.rowDim(); to++)
      {
         for (unsigned int from = 0; from < layer_weight_learning_rates.colDim();
               from++)
         {
            layer_weight_learning_rates.at(to, from) *= _reductionFactor;
         }
      }
   }
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_LRATE_POLICY_H_ */
