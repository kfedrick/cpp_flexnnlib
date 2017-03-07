/*
 * NeuralNetwork.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: kfedrick
 */

#include <sstream>
#include <iostream>
#include <algorithm>

#include "BaseNeuralNet.h"

using namespace std;
namespace flex_neuralnet
{

BaseNeuralNet::BaseNeuralNet(const char* _name) :
      NamedObject(_name)
{
   topology_initialized_flag = false;
   recurrent_network_flag = false;
   max_closed_loop_steps = 0;

   resize_history(2);
}

BaseNeuralNet::BaseNeuralNet(const string& _name = "BaseNeuralNet") :
      NamedObject(_name)
{
   topology_initialized_flag = false;
   recurrent_network_flag = false;
   max_closed_loop_steps = 0;

   resize_history(2);
}

BaseNeuralNet::~BaseNeuralNet()
{
}

void BaseNeuralNet::resize_history(unsigned int sz)
{
   recur_history_size = sz;

   for (int i = 0; i < layer_activation_order.size(); i++)
   {
      // Get a network layer
      BaseLayer* layer = layer_activation_order[i];

      // Get the LayerInputFacade for this network layer from the connection map
      map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
            layer_input_conn_map.find(layer);
      ConnectionMap* layer_input = map_entry->second;

      // Activate the network layer with the raw input from it's layer input connection map
      layer->resize_history(recur_history_size);
   }

   closed_loop_steps.resize(recur_history_size);
   for (unsigned int ndx = 0; ndx < recur_history_size; ndx++)
      closed_loop_steps.at(ndx) = 0;
}

void BaseNeuralNet::set_max_closedloop_steps(unsigned int steps)
{
   max_closed_loop_steps = steps;
}

unsigned int BaseNeuralNet::get_output_size() const
{
   return network_output_map.size();
}

/*
 void NeuralNet::install_input_layer(BaseLayer* layer)
 {
 input_layers.push_back(layer);
 layer_input_dependency_map[layer] = set<const BaseLayer*>();
 }
 */

void BaseNeuralNet::install_hidden_layer(BaseLayer* layer)
{
   network_layers.push_back(layer);
   layer_input_conn_map[layer] = new ConnectionMap(*layer);
   layer_input_dependency_map[layer] = set<const BaseLayer*>();
}

void BaseNeuralNet::install_output_layer(BaseLayer* layer)
{
   network_layers.push_back(layer);
   output_layers.push_back(layer);
   network_output_map.connect(*layer);

   layer_input_conn_map[layer] = new ConnectionMap(*layer);
   layer_input_dependency_map[layer] = set<const BaseLayer*>();
}

void BaseNeuralNet::connect(const Pattern& ipattern, unsigned int patternNdx,
      BaseLayer& to)
{
   // Find the connection target layer in the input facade map
   map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
         layer_input_conn_map.find(&to);
   if (map_entry == layer_input_conn_map.end())
      throw invalid_argument(
            "NeuralNet::connect(InputLayer) : Target layer not found in network.");

   // Add the 'from' layer as an input connection to the corresponding LayerInputFacade
   map_entry->second->connect(ipattern, patternNdx);

   // Resize the weight array for the 'to' layer
   to.resize_input_vector(map_entry->second->size());

   // Resize network input error vector (possibly redundant)
   network_input_error.clear();
   for (unsigned int i = 0; i < ipattern.size(); i++)
      network_input_error.push_back(ipattern.at(i));

   topology_initialized_flag = false;

   update_topology();
}

void BaseNeuralNet::connect(BaseLayer& from, BaseLayer& to)
{
   map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
         layer_input_conn_map.find(&to);
   if (map_entry == layer_input_conn_map.end())
      throw invalid_argument(
            "NeuralNet::connect() : Target layer not found in network.");

   vector<BaseLayer*>::iterator net_layer_itr = find(network_layers.begin(),
         network_layers.end(), &from);
   if (net_layer_itr == network_layers.end())
      throw invalid_argument(
            "NeuralNet::connect() : Input layer not found in network.");

   // Add the 'from' layer as an input connection to the ConnectionMap into the 'to' layer
   map_entry->second->connect(from);

   // Resize the weight array for the 'to' layer
   to.resize_input_vector(map_entry->second->size());

   topology_initialized_flag = false;

   update_topology();
}

const BaseLayer& BaseNeuralNet::layer(int index)
{
   return *network_layers.at(index);
}

const Pattern& BaseNeuralNet::operator()(void)
{
   if (!topology_initialized_flag)
      update_topology();

   return network_output_pattern;
}

const Pattern& BaseNeuralNet::operator()(const Pattern& ipattern,
      unsigned int timeStep)
{

   if (!topology_initialized_flag)
      update_topology();

   /*
    * Iterate through all recurent steps
    */

   closed_loop_steps.at(timeStep) = 0;
   for (unsigned int closed_loop_step = 0;
         closed_loop_step < max_closed_loop_steps + 1; closed_loop_step++)
   {
      /*
       * Activate all network layers
       */
      for (int i = 0; i < layer_activation_order.size(); i++)
      {
         // Get a network layer
         BaseLayer* layer = layer_activation_order[i];

         // Get the LayerInputFacade for this network layer from the connection map
         map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
               layer_input_conn_map.find(layer);
         ConnectionMap* layer_input = map_entry->second;

         // Activate the network layer with the raw input from it's layer input connection map
         const vector<double>& invec = (*layer_input)(ipattern, timeStep,
               closed_loop_step);

         layer->activate(invec, timeStep);
      }

      closed_loop_steps.at(timeStep)++;
   }

      // One of the activation steps was the base step, not a closed loop step
   closed_loop_steps.at(timeStep)--;

   network_output_pattern
   = network_output_map(ipattern, timeStep, 0);

   return network_output_pattern;
}

const PatternSequence& BaseNeuralNet::operator()(
      const PatternSequence& ipattseq, unsigned int startTimeStep)
{
   if (!topology_initialized_flag)
      update_topology();

   if (recurrent_network_flag)
      resize_history(startTimeStep + ipattseq.size());

   network_output_patternseq.clear();
   network_output_patternseq.resize(ipattseq.size());

   /*
    * Iterate through all the Patterns in this sequence
    */
   unsigned int time_step = startTimeStep;
   for (unsigned int pattern_ndx = 0; pattern_ndx < ipattseq.size();
         pattern_ndx++)
   {
      const Pattern& ipatt = ipattseq.at(pattern_ndx);

      /*
       * Iterate through all closed loop steps
       */
      closed_loop_steps.at(time_step) = 0;
      for (unsigned int closed_loop_step = 0;
            closed_loop_step < max_closed_loop_steps + 1; closed_loop_step++)
      {
         /*
          * Activate all network layers
          */
         for (unsigned int i = 0; i < layer_activation_order.size(); i++)
         {
            // Get a network layer
            BaseLayer* layer = layer_activation_order[i];

            // Get the LayerInputFacade for this network layer from the connection map
            map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
                  layer_input_conn_map.find(layer);
            ConnectionMap* layer_input = map_entry->second;

            // Activate the network layer with the raw input from it's layer input connection map
            const vector<double>& invec = (*layer_input)(ipatt, startTimeStep,
                  closed_loop_step);
            layer->activate(invec, time_step);
         }

         closed_loop_steps.at(time_step)++;}

         // One of the activation steps was the base step, not a closed loop step
      closed_loop_steps.at(time_step)--;

      network_output_pattern
      = network_output_map(ipatt, time_step, 0);
      network_output_patternseq.at(pattern_ndx) = network_output_pattern;

      if (recurrent_network_flag)
         time_step++;
   }

   return network_output_patternseq;
}

const vector<BaseLayer*> BaseNeuralNet::get_network_layers() const
{
   return network_layers;
}

void BaseNeuralNet::backprop(const vector<double>& isse, unsigned int timeStep)
{
   /*
    * Backprop network error to output layers
    */
   const vector<vector<double> >& network_errorv = network_output_map.get_error(
         isse);

   const vector<ConnectionEntry>& network_output_connvec =
         network_output_map.get_input_connections();
   for (int conn_ndx = 0; conn_ndx < network_output_connvec.size(); conn_ndx++)
   {
      const ConnectionEntry& conn = network_output_connvec.at(conn_ndx);
      BaseLayer& layer = conn.get_input_layer();

      layer.backprop(network_errorv.at(conn_ndx), timeStep);
   }

   // cout << "closed loop steps for time " << timeStep << " = " << closed_loop_steps.at(timeStep) << endl;
   bool done = false;
   for (unsigned int closed_loop_step = closed_loop_steps.at(timeStep); !done;
         closed_loop_step--)
   {
      /*
       * Backprop error through network in reverse order of the layer activation ordering
       */
      for (int i = layer_activation_order.size() - 1; i >= 0; i--)
      {
         BaseLayer& layer = *layer_activation_order[i];

         // Backprop error through this layer to the layer inputs
         layer.backprop(timeStep);

         // backprop error at the inputs to layers providing input to this layer
         backprop_scatter(layer, timeStep, closed_loop_step);
      }

      if (closed_loop_step == 0)
         done = true;
   }
}

void BaseNeuralNet::backprop_scatter(BaseLayer& layer, unsigned int timeStep,
      unsigned int closedLoopStep)
{
   map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
         layer_input_conn_map.find(&layer);
   ConnectionMap* layer_input = map_entry->second;

   const vector<vector<double> >& errorv = layer_input->get_error(timeStep);
   const vector<ConnectionEntry>& connvec =
         layer_input->get_input_connections();

   for (int conn_ndx = 0; conn_ndx < connvec.size(); conn_ndx++)
   {
      const ConnectionEntry& conn = connvec.at(conn_ndx);

      if (conn.is_input_connection())
      {
         unsigned int ipatt_ndx = conn.get_input_pattern_index();
         vector<double>& errvec = network_input_error.at(ipatt_ndx);

         for (unsigned int vec_ndx = 0; vec_ndx < errvec.size(); vec_ndx++)
            errvec.at(vec_ndx) += errorv.at(conn_ndx).at(vec_ndx);
         continue;
      }

      unsigned int bpTimeStep = timeStep;
      if (conn.is_recurrent() && closedLoopStep == 0)
         bpTimeStep--;

      BaseLayer& layer = conn.get_input_layer();
      /*
       cout << "backprop to " << layer.name() << endl;
       for (unsigned int i=0; i<errorv.at(conn_ndx).size(); i++)
       cout << errorv.at(conn_ndx).at(i) << " ";
       cout << endl << "*******" << endl;
       */
      layer.backprop(errorv.at(conn_ndx), bpTimeStep);
   }
}

void BaseNeuralNet::clear_error(unsigned int timeStep)
{
   network_output_map.clear_error();

   /*
    * make all network layers clear their error vectors
    */
   for (int i = layer_activation_order.size() - 1; i >= 0; i--)
   {
      BaseLayer& layer = *layer_activation_order[i];
      layer.clear_error(timeStep);
   }

   for (unsigned int ierr_ndx = 0; ierr_ndx < network_input_error.size();
         ierr_ndx++)
   {
      vector<double>& errvec = network_input_error.at(ierr_ndx);
      errvec.assign(errvec.size(), 0.0);
   }
}

const vector<vector<double> >& BaseNeuralNet::get_input_error(
      unsigned int timeStep) const
{
   return network_input_error;
}

const NetworkWeightsData& BaseNeuralNet::get_network_weights()
{
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      LayerWeightsData& layer_weights_data =
            network_weights_data.new_layer_weights(name);

      // Layer value at time step 0 is the initial layer value
      layer_weights_data.initial_value = layer(0);

      layer_weights_data.biases = layer.get_biases();

      const Array<double>& layer_weights = layer.get_weights();
      layer_weights_data.weights.resize(layer_weights.rowDim(),
            layer_weights.colDim());
      layer_weights_data.weights = layer.get_weights();
   }

   return network_weights_data;
}

void BaseNeuralNet::update_topology()
{
   update_layer_input_dependencies();
   update_recurrent_connection_flags();
   update_activation_order();

   topology_initialized_flag = true;
}

void BaseNeuralNet::update_layer_input_dependencies()
{
//   cout << "NeuralNet::update_layer_input_dependencies()" << endl;

   set<BaseLayer*> dependencies;
   for (int i = 0; i < network_layers.size(); i++)
   {
      set<const BaseLayer*>& dependencies = layer_input_dependency_map.at(
            network_layers[i]);
      dependencies.clear();

      calc_layer_input_dependencies(dependencies, network_layers[i]);
      /*
       cout << "*** dependencies for " << network_layers[i]->name() << endl;
       for (set<const BaseLayer*>::iterator itr=dependencies.begin(); itr != dependencies.end(); itr++)
       cout << (*itr)->name() << endl;
       cout << "******************\n" << endl;
       */
   }
}

/*
 * Return a set of pointers to layers that feed activity to this layer directly or
 * indirectly. That is to say all layers for which there is a path in the directed graph
 * from that layer to the specified layer. Include the specified layer itself in the set
 * in the event there is recurency.
 */
void BaseNeuralNet::calc_layer_input_dependencies(
      set<const BaseLayer*>& dependencies, const BaseLayer* layer)
{
   map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
         layer_input_conn_map.find(layer);

   /*
    * If we can't find an entry for this layer in the layer connection map then there's a problem.
    */
   if (map_entry == layer_input_conn_map.end())
      throw invalid_argument(
            "NeuralNet::calc_dependencies() : Target layer not found in network.");

   ConnectionMap* lif = map_entry->second;
   const vector<ConnectionEntry>& input_conn = lif->get_input_connections();
//   cout << "NEW layer input map size " << layer->name() << " = " << input_conn.size() << endl;

// Add indirect dependencies by recursing on direct dependencies not already in dependency set
   for (int i = 0; i < input_conn.size(); i++)
   {

      // Ignore connections from network inputs
      if (input_conn[i].is_input_connection())
         continue;

      if (dependencies.find(&input_conn[i].get_input_layer())
            == dependencies.end())
      {
         dependencies.insert(&input_conn[i].get_input_layer());
         calc_layer_input_dependencies(dependencies,
               &input_conn[i].get_input_layer());
      }
   }
}

void BaseNeuralNet::update_recurrent_connection_flags()
{
   recurrent_network_flag = false;

   for (int i = 0; i < network_layers.size(); i++)
   {
      BaseLayer* to_layer = network_layers[i];

      map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
            layer_input_conn_map.find(to_layer);
      if (map_entry == layer_input_conn_map.end())
         throw invalid_argument(
               "NeuralNet::update_recurrent_connection_flags() : layer not found in network.");

      ConnectionMap* lif = map_entry->second;
      vector<ConnectionEntry>& input_conn = lif->get_input_connections();

      /*
       * First pass detect recurrent dependencies
       */
      bool is_recurrent_path[input_conn.size()];
      bool all_recurrent = true;
      for (int j = 0; j < input_conn.size(); j++)
      {
         if (input_conn[j].is_input_connection())
         {
            is_recurrent_path[j] = false;
         }
         else
         {
            const BaseLayer& input_layer = input_conn[j].get_input_layer();
            set<const BaseLayer*>& dependencies = layer_input_dependency_map.at(
                  &input_layer);

            is_recurrent_path[j] = false;
            if (dependencies.find(to_layer) != dependencies.end())
               is_recurrent_path[j] = true;
         }

         all_recurrent &= is_recurrent_path[j];
      }

      /*
       * Second pass, remove recurrent flag where *all* input connections are recurrent
       * Only mark those recurrent where there are other non-recurrent paths
       */
      for (int j = 0; j < input_conn.size(); j++)
      {
         if (all_recurrent)
            input_conn[j].set_recurrent(false);
         else
         {
            if (is_recurrent_path[j])
            {
               input_conn[j].set_recurrent(true);
               recurrent_network_flag = true;
            }
            else
               input_conn[j].set_recurrent(false);
         }

         /*
          if (input_conn[j].is_recurrent())
          cout << "recurrent conn from " << input_conn[j].get_input_layer().name() << " to " << to_layer->name() << endl;
          else
          {
          if (input_conn[j].is_input_connection())
          cout << "non-recurrent conn from input pattern " << input_conn[j].get_input_pattern_index() << endl;
          else
          cout << "non-recurrent conn from " << input_conn[j].get_input_layer().name() << " to " << to_layer->name() << endl;
          }
          */
      }
   }
}

void BaseNeuralNet::update_activation_order()
{
   layer_activation_order.clear();
   for (int i = 0; i < network_layers.size(); i++)
   {
      vector<BaseLayer*>::iterator itr = layer_activation_order.begin();
      while (itr != layer_activation_order.end())
      {
         if (requires(*itr, network_layers[i]))
            break;
         else
            itr++;
      }
      layer_activation_order.insert(itr, network_layers[i]);
   }
}

/*
 * Returns true of layer1 has a non-recursive dependency on layer2. This
 * indicates that layer2 must be activated before layer1.
 */
bool BaseNeuralNet::requires(const BaseLayer* layer1, const BaseLayer* layer2)
{
   set<const BaseLayer*> layer1_dependencies;
   calc_layer_feedforward_dependencies(layer1_dependencies, layer1);

   //cout << "layer dep size " << layer1->name() << " = " << layer1_dependencies.size() << endl;
   if (layer1_dependencies.find(layer2) != layer1_dependencies.end())
      return true;
   else
      return false;
}

/*
 * Return a set of pointers to layers that feed activity to this layer directly or
 * indirectly. That is to say all layers for which there is a path in the directed graph
 * from that layer to the specified layer. Include the specified layer itself in the set
 * in the event there is recurrency.
 */
void BaseNeuralNet::calc_layer_feedforward_dependencies(
      set<const BaseLayer*>& dependencies, const BaseLayer* layer)
{
   map<const BaseLayer*, ConnectionMap*>::iterator map_entry =
         layer_input_conn_map.find(layer);

   if (map_entry == layer_input_conn_map.end())
      throw invalid_argument(
            "NeuralNet::calc_layer_feedforward_dependencies() : Layer not found in network.");

   ConnectionMap* lif = map_entry->second;
   vector<ConnectionEntry>& input_conn = lif->get_input_connections();
   //cout << "layer input map size " << layer->name() << " = " << input_conn.size() << endl;

   /*
    * Add each connected layer to the set of input dependencies if it isn't already in the set
    * and recursively retrieve it's dependencies. Recurse breadth first so it terminates faster
    * when if it finds a recurrent path back to itself.
    */
   for (int i = 0; i < input_conn.size(); i++)
   {
      if (!input_conn[i].is_recurrent() && !input_conn[i].is_input_connection())
      {
         dependencies.insert(&input_conn[i].get_input_layer());
         calc_layer_feedforward_dependencies(dependencies,
               &input_conn[i].get_input_layer());
      }
   }
}

} /* namespace flex_neuralnet */

