/*
 * NeuralNetwork.h
 *
 *  Created on: Feb 4, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_BASENET_H_
#define FLEX_NEURALNET_BASENET_H_

#include <vector>
#include <map>
#include <set>
#include <stdexcept>

#include "Array.h"
#include "BaseLayer.h"
#include "Layer.h"
#include "ConnectionMap.h"
#include "Pattern.h"
#include "PatternSequence.h"
#include "Exemplar.h"
#include "OutputErrorFunctor.h"
#include "SumSquaredError.h"
#include "NetworkWeightsData.h"

using namespace std;

namespace flex_neuralnet
{


class BaseNeuralNet : public flex_neuralnet::NamedObject
{

public:
   BaseNeuralNet(const char* _name  = "BaseNeuralNet");
   BaseNeuralNet(const string& _name);
   virtual ~BaseNeuralNet();

   /*
    * Resize the activation history capacity for the network
    */
   virtual void resize_history(unsigned int sz = 2);

   void set_max_closedloop_steps(unsigned int steps);

   /* *********************************************
    *    Network factory methods
    * *********************************************
    */

   /*
    * Add a new hidden layer with net input function _NetInFunc and transfer function _TransFunc
    */

   template <class _NetInFunc, class _TransFunc>
   Layer<_NetInFunc, _TransFunc>& new_hidden_layer(unsigned int sz, const char* name = 0);

   template <class _NetInFunc, class _TransFunc>
   Layer<_NetInFunc, _TransFunc>& new_output_layer(unsigned int sz, const char* name = 0);

   void connect(BaseLayer& from, BaseLayer& to);
   void connect(const Pattern& ipattern, unsigned int patternNdx, BaseLayer& to);


   /* *********************************************
    *    Network accessor functions
    * *********************************************
    */
   const BaseLayer& layer(int ndx);

   unsigned int get_output_size() const;

   /* *********************************************
    *    Network simulation functions
    * *********************************************
    */

   const Pattern& operator()();
   const Pattern& operator()(const Pattern& ipattern, unsigned int recurStep = 1);
   const PatternSequence& operator()(const PatternSequence& ipattern, unsigned int recurStep = 1);

   const vector<BaseLayer*> get_network_layers() const;

   NetworkWeightsData& get_network_weights();

   vector< vector<double> >& get_input_error(unsigned int timeStep = 1);

   void backprop(const vector<double>& isse, unsigned int timeStep = 1);
   void backprop_scatter(BaseLayer& layer, unsigned int timeStep = 1, unsigned int closedLoopStep = 0);

   const vector<BaseLayer*>& get_layer_activation_order();
   ConnectionMap& get_network_output_connection_map();
   ConnectionMap* get_layer_connection_map(const BaseLayer& layer);

   void clear_error(unsigned int timeStep = 1);

private:

   /* *********************************************
    *    Network objects
    * *********************************************
    */
   vector<BaseLayer*> network_layers;
   vector<BaseLayer*> output_layers;

   /* *********************************************
    *    Network topology
    * *********************************************
    */
   bool recurrent_network_flag;

   vector< vector<double> > network_input_error;

   /*
    * The ConnectionMap to coalesce all output layers in the network
    */
   ConnectionMap network_output_map;
   Pattern network_output_pattern;
   PatternSequence network_output_patternseq;

   map<const BaseLayer*, ConnectionMap* > layer_input_conn_map;

   map<const BaseLayer*, set<const BaseLayer*> > layer_input_dependency_map;

   vector<BaseLayer*> layer_activation_order;

   OutputErrorFunctor* oerr;

   unsigned int recur_history_size;

   /* **********************************************
    *    Activation state information
    */
   bool topology_initialized_flag;

   unsigned int max_closed_loop_steps;
   vector<unsigned int> closed_loop_steps;

   unsigned int last_activation_recur_steps;

   NetworkWeightsData network_weights_data;

private:

   void install_hidden_layer(BaseLayer* layer);
   void install_output_layer(BaseLayer* layer);

   void update_topology();

   void update_layer_input_dependencies();

   void calc_layer_input_dependencies(set<const BaseLayer*>& dependencies, const BaseLayer* layer);

   void update_recurrent_connection_flags();

   void update_activation_order();

   /*
    * Returns true of layer1 has a non-recursive dependency on layer2. This
    * indicates that layer2 must be activated before layer1.
    */
   bool requires(const BaseLayer* layer1, const BaseLayer* layer2);

   /*
    * Returns an ordered vector of all the layers that are in the specified layers
    * direct dependency tree (i.e. all layers in the activation chain that provide
    * input to the specified layer)
    */
   void calc_layer_feedforward_dependencies(set<const BaseLayer*>& dependencies, const BaseLayer* layer);
};



template <class _NetInFunc, class _TransFunc> inline
Layer<_NetInFunc, _TransFunc>& BaseNeuralNet::new_hidden_layer(unsigned int _sz, const char* _name)
{
   stringstream layer_name;
   if (_name == 0)
      layer_name << "hidden-layer-" << network_layers.size();
   else
      layer_name << _name;

   Layer<_NetInFunc, _TransFunc>* layer = new Layer<_NetInFunc, _TransFunc>(_sz, layer_name.str());
   install_hidden_layer(layer);

   return *layer;
}

template <class _NetInFunc, class _TransFunc> inline
Layer<_NetInFunc, _TransFunc>& BaseNeuralNet::new_output_layer(unsigned int _sz, const char* _name)
{
   stringstream layer_name;
   if (_name == 0)
      layer_name << "output-layer-" << output_layers.size();
   else
      layer_name << _name;

   Layer<_NetInFunc, _TransFunc>* layer = new Layer<_NetInFunc, _TransFunc>(_sz, layer_name.str());
   install_output_layer(layer);

   return *layer;
}

inline
ConnectionMap& BaseNeuralNet::get_network_output_connection_map()
{
   return network_output_map;
}

inline
ConnectionMap* BaseNeuralNet::get_layer_connection_map(const BaseLayer& layer)
{
   map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
         layer_input_conn_map.find(&layer);
   ConnectionMap* layer_input = map_entry->second;

   return layer_input;
}

inline
const vector<BaseLayer*>& BaseNeuralNet::get_layer_activation_order()
{
   return layer_activation_order;
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_BASENET_H_ */
