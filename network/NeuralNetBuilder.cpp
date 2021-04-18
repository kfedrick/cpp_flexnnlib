//
// Created by kfedrick on 4/10/21.
//

#include "NeuralNetBuilder.h"

using flexnnet::NeuralNetBuilder;

NeuralNetBuilder::NeuralNetBuilder()
{
}

NeuralNetBuilder::NeuralNetBuilder(const ValarrMap& _xinput_sample)
{
   sample_external_input = _xinput_sample;
}

NeuralNetBuilder::~NeuralNetBuilder(void)
{

}

/**
 * Add a connection to the basic_layer, _to, from the basic_layer, _from.
 *
 * @param _to - the name of the basic_layer to recieve input
 * @param _from - the name of the basic_layer to send its output
 */
void
NeuralNetBuilder::add_layer_connection(const std::string& _to, const std::string& _from, LayerConnRecord::ConnectionType _type)
{
   /*
    * Check that _to and _from layers exist in the list of network layers.
    * If not throw an exception.
    */
   if (topo_layers.find(_to) == topo_layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NeuralNetBuilder::add_layer_connection() - "
           << "Target layer : \"" << _to << "\" does not exist.\n";
      throw std::invalid_argument(sout.str());
   }

   if (topo_layers.find(_from) == topo_layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NeuralNetBuilder::add_layer_connection() - "
           << "Layer does not exist, _from = \"" << _from << "\".\n";
      throw std::invalid_argument(sout.str());
   }

   // Get from_dependencies so we can validate connection type can be made
   std::set<std::string> from_dependencies;
   std::set<std::string> to_dependencies;
   get_input_dependencies(from_dependencies, _from);
   get_input_dependencies(to_dependencies, _to);

   switch (_type)
   {
      case LayerConnRecord::Forward:

         // Check whether a forward connection is allowable.
         validate_forward_connection(_to, _from, from_dependencies);

         // If we got here it's OK to add forward connection
         topo_layers[_to]->add_connection("activation", topo_layers[_from], _type);
         topo_layers[_from]->add_connection("backprop", topo_layers[_to], _type);
         break;

      case LayerConnRecord::Recurrent:

         // Check whether a recurrent connection is allowable.
         validate_recurrent_connection(_to, _from, from_dependencies);

         // If we got here it's OK to add recurrent connection
         topo_layers[_to]->add_connection("activation", topo_layers[_from], _type);
         topo_layers[_from]->add_connection("backprop", topo_layers[_to], _type);
         break;

      case LayerConnRecord::Lateral:

         // Check whether a lateral connection is allowable.
         validate_lateral_connection(_to, _from, to_dependencies, from_dependencies);

         // If we got here it's OK to add lateral connections
         topo_layers[_to]->add_connection("activation", topo_layers[_from], _type);
         topo_layers[_from]->add_connection("backprop", topo_layers[_to], _type);

         topo_layers[_from]->add_connection("activation", topo_layers[_to], _type);
         topo_layers[_to]->add_connection("backprop", topo_layers[_from], _type);
         break;
   };

   // Always update the activation order after adding a new connection
   update_activation_order();
}

void
NeuralNetBuilder::validate_forward_connection(const std::string& _to, const std::string& _from, const std::set<
   std::string>& _from_dependencies)
{
   /*
    * In order to add a valid forward connection from basic_layer _from to basic_layer _to, the
    * _from basic_layer must not have a forward activation dependency on output from the
    * _to basic_layer as this would indicate a cycle. Likewise the _to and _from layers
    * must not be the same as this would cause a cycle.
    */
   if (_to == _from || _from_dependencies.find(_to) != _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NeuralNetBuilder::validate_forward_connection() - "
           << "Can't add Forward connection from  \""
           << _from << "\" => \"" << _to << "\" ( "
           << " would create cycle)." << std::endl;
      throw std::invalid_argument(sout.str());
   }
}

void
NeuralNetBuilder::validate_recurrent_connection(const std::string& _to, const std::string& _from, const std::set<
   std::string>& _from_dependencies)
{
   /*
    * To add a valid recurrent connection from basic_layer _from to basic_layer _to, the
    * _from basic_layer must already have a forward activation dependency from the
    * _to basic_layer, or the _to and _from basic_layer must be the same.
    */
   if (_to != _from && _from_dependencies.find(_to) == _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NeuralNetBuilder::validate_recurrent_connection() - "
           << "Can't add Recurrent connection from  \""
           << _from << "\" => \"" << _to << "\" - "
           << " no forward depenencies." << std::endl;
      throw std::invalid_argument(sout.str());
   }
}

void
NeuralNetBuilder::validate_lateral_connection(const std::string& _to, const std::string& _from, std::set<std::string>& _to_dependencies, std::set<std::string>& _from_dependencies)
{
   /*
    * To add a valid lateral connection from basic_layer _from to basic_layer _to, the
    * _to and _from basic_layer must be distinct and there must not be any existing
    * forward connection from either one to the other.
    */
   if (_to == _from)
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NeuralNetBuilder::validate_lateral_connection() - "
           << "Can't add Lateral connection from  \""
           << _from << "\" to itself." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   if (_to_dependencies.find(_from) != _to_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NeuralNetBuilder::validate_lateral_connection() - "
           << "Can't add Lateral connection from  \""
           << _from << "\" => \"" << _to << "\" - "
           << " Forward connection already exist." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   if (_from_dependencies.find(_to) != _from_dependencies.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : NeuralNetBuilder::validate_lateral_connection() - "
           << "Can't add Lateral connection from  \""
           << _to << "\" => \"" << _from << "\" - "
           << " Forward connection already exist." << std::endl;
      throw std::invalid_argument(sout.str());
   }
}

/**
 * Add a connection to the basic_layer, _to, from an external input vector.
 * @param _to
 * @param _vec
 */
void
NeuralNetBuilder::add_external_input_connection(const std::string& _to, const std::string& _field)
{
   /*
    * Check that _to basic_layer and external input field, _field, exist in the
    * list of network layers and in the external input sample, respectively.
    * If not throw an exception.
    */
   if (topo_layers.find(_to) == topo_layers.end())
   {
      static std::stringstream sout;
      sout << "Error : NeuralNetBuilder::add_external_input_connection() - "
           << "Target layer : \"" << _to << "\" does not exist.\n";
      throw std::invalid_argument(sout.str());
   }

   if (sample_external_input.find(_field) == sample_external_input.end())
   {
      static std::stringstream sout;
      sout << "Error : NeuralNetBuilder::add_external_input_connection() - "
           << "External input field : \"" << _field.c_str()
           << "\" does not exist.\n";
      throw std::invalid_argument(sout.str());
   }

   topo_layers[_to]->add_external_input_field(_field, sample_external_input.at(_field).size());

   // Always update the activation order after adding a new connection
   update_activation_order();
}

void NeuralNetBuilder::update_activation_order(void)
{
   topo_ordered_layers.clear();
   for (auto& item : topo_layers)
   {
      std::string layer_name = item.first;

      /*
       * Find the first (if any) layer already in the ordered activation list
       * that has a feedforward input dependency on the new layer and insert the
       * new layer just in front of it.
       */
      bool inserted = false;
      for (auto it = topo_ordered_layers.begin(); it != topo_ordered_layers.end(); ++it)
      {
         auto& ordered_layer_name = (*it)->name();

         // Get dependencies for ordered_layer_name
         std::set<std::string> dependencies;
         get_input_dependencies(dependencies, ordered_layer_name);

         // If new layer_name is in the list then break now and insert.
         if (dependencies.find(layer_name) != dependencies.end())
         {
            topo_ordered_layers.insert(it, topo_layers[layer_name]);
            inserted = true;
            break;
         }
      }

      // If it wasn't already inserted, add it to the end now
      if (!inserted)
         topo_ordered_layers.push_back(topo_layers[layer_name]);
   }
}

/**
 * Return a set containing the names of layers connected to the basic_layer, _to, through
 * a chain of one or more forward connections.
 *
 * @param _dependencies
 * @param _name
 */
void NeuralNetBuilder::get_input_dependencies(std::set<std::string>& _dependencies, const std::string& _from)
{
   const std::vector<LayerConnRecord>& activation_conn_list = topo_layers[_from]->activation_connections;

// Add indirect dependencies by recursing on direct dependencies not already in dependency set_weights
   for (auto& record : activation_conn_list)
   {
      // Ignore recurrent connections
      if (record.is_recurrent())
         continue;

      /*
       * If this basic_layer is not already in our dependency list then
       * add it and recurse; otherwise we've visited this basic_layer already
       * so do nothing.
       */
      if (_dependencies.find(record.layer().name()) == _dependencies.end())
      {
         _dependencies.insert(record.layer().name());
         get_input_dependencies(_dependencies, record.layer().name());
      }
   }
}

void NeuralNetBuilder::alloc_layer_data(void)
{
   for (auto& it : topo_layers)
   {
      const std::string& layer_id = it.first;
      size_t layer_sz = it.second->size();
      unsigned int layer_input_sz = it.second->calc_input_size();

      // Set input size for basic layer so it resizes the weights
      it.second->set_input_size(layer_input_sz);
   }
}