//
// Created by kfedrick on 5/28/19.
//

#include "BasicNeuralNetFactory.h"
#include "OldNetworkLayer.h"

using flexnnet::BasicNeuralNetFactory;
using flexnnet::BasicNeuralNet;

BasicNeuralNetFactory::BasicNeuralNetFactory()
{
   clear();
}

BasicNeuralNetFactory::~BasicNeuralNetFactory()
{
   clear();
}

/**
 * Clear all references to object generated in the process of
 * building a network, but do not delete the objects (layers and etc.)
 * dynamically generated.
 */
void BasicNeuralNetFactory::clear(void)
{
   built = false;
   layer_external_input_set = false;
   network_input_set = false;
   recurrent_network_flag = false;
   layer_activation_order.clear();
   layers.clear();
}

void BasicNeuralNetFactory::set_network_input(const Datum& _network_input)
{
   if (layer_external_input_set)
   {
      static std::stringstream sout;
      sout.clear();
      sout
         << "Error : BasicNeuralNetFactory::set_network_input() - "
         << "Can't reset network input after basiclayer external inputs have been set_weights."
         << std::endl;
      throw std::logic_error(sout.str());
   }

   network_input = _network_input;
   network_input_set = true;
}

std::shared_ptr<BasicNeuralNet> BasicNeuralNetFactory::build(const std::string& _network_name)
{
   updateActivationOrder();

   validate();

   // Create a list of basiclayer pointers sorted by activation order.
   std::vector<std::shared_ptr<OldNetworkLayer>> ordered_layer_list;
   for (auto& layer_name : layer_activation_order)
   {
      ordered_layer_list.push_back(layers[layer_name]);
   }

   // Construct the network
   auto net = std::shared_ptr<BasicNeuralNet>(new BasicNeuralNet(ordered_layer_list, recurrent_network_flag, _network_name));

   // Mark the network built, clear local pointers and return it to the caller.
   built = true;
   clear();

   return net;
}

/**
 * Add connection to the basiclayer, _to, from the basiclayer, _from.
 *
 * @param _to - the name of the basiclayer to recieve input
 * @param _from - the name of the basiclayer to send its output
 */
void
BasicNeuralNetFactory::add_layer_connection(const std::string& _to, const std::string& _from, OldLayerConnRecord::ConnectionType _type)
{
   if (layers.find(_to) == layers.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : BasicNeuralNetFactory::add_layer_connection() - "
           << "No such basiclayer, _to = \"" << _to.c_str() << "\"." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   if (layers.find(_from) == layers.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : BasicNeuralNetFactory::add_layer_connection() - "
           << "No such basiclayer, _from = \"" << _from.c_str() << "\"." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   OldNetworkLayer& to = *layers[_to];
   OldNetworkLayer& from = *layers[_from];

   // Get dependencies so we can validate connection type can be made
   std::set<std::string> dependencies;
   getForwardDependencies(dependencies, _from);

   size_t to_input_sz;
   size_t from_input_sz;

   switch (_type)
   {
      /*
       * In order to set_weights a valid forward connection from basiclayer _from to basiclayer _to, the
       * _from basiclayer must not have a forward activation dependency on output from the
       * _to basiclayer as this would indicate a cycle. Likewise the _to and _from layers
       * must not be the same.
       */
      case OldLayerConnRecord::Forward:

         if (_to == _from || dependencies.find(_to) != dependencies.end())
         {
            static std::stringstream sout;
            sout.clear();
            sout << "Error : BasicNeuralNetFactory::add_connection() - Can't add Forward connection from  \""
                 << _from.c_str() << "\" => \"" << _to.c_str() << "\" - "
                 << " would create cycle." << std::endl;
            throw std::invalid_argument(sout.str());
         }

         to_input_sz = to.add_connection(from, _type);
         to.resize_input(to_input_sz);
         break;

         /*
          * To set_weights a valid recurrent connection from basiclayer _from to basiclayer _to, the
          * _from basiclayer must already have a forward activation depencency from the
          * _to basiclayer, or the _to and _from basiclayer must be the same.
          */
      case OldLayerConnRecord::Recurrent:
         std::cout << _to.c_str() << " in set_weights? " << (dependencies.find(_to) != dependencies.end()) << std::endl;

         if (_to != _from && dependencies.find(_to) == dependencies.end())
         {
            static std::stringstream sout;
            sout.clear();
            sout << "Error : BasicNeuralNetFactory::add_connection() - "
                 << "Can't add Recurrent connection from  \""
                 << _from.c_str() << "\" => \"" << _to.c_str() << "\" - "
                 << " no forward depenencies." << std::endl;
            throw std::invalid_argument(sout.str());
         }

         to_input_sz = to.add_connection(from, _type);
         to.resize_input(to_input_sz);
         recurrent_network_flag = true;
         break;

         /*
          * To set_weights a valid lateral connection from basiclayer _from to basiclayer _to, the
          * _to and _from basiclayer must be distinct.
          */
      case OldLayerConnRecord::Lateral:std::set<std::string> d1, d2;
         if (_to == _from)
         {
            static std::stringstream sout;
            sout.clear();
            sout << "Error : BasicNeuralNetFactory::add_connection() - "
                 << "Can't add Lateral connection from  \""
                 << _from.c_str() << "\" to itself." << std::endl;
            throw std::invalid_argument(sout.str());
         }

         to_input_sz = to.add_connection(from, _type);
         from_input_sz = from.add_connection(from, _type);

         to.resize_input(to_input_sz);
         from.resize_input(from_input_sz);

         break;
   }
}

/**
 * Add a connection to the basiclayer, _to, from an external input vector.
 * @param _to
 * @param _vec
 */
void
BasicNeuralNetFactory::set_layer_external_input(const std::string& _to, const Datum& _network_input, const std::set<std::string>& _indexSet)
{
   if (layers.find(_to) == layers.end())
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : BasicNeuralNetFactory::add_layer_external_input() - "
           << "No such basiclayer, _to = \"" << _to.c_str() << "\"." << std::endl;
      throw std::invalid_argument(sout.str());
   }

   if (!network_input_set)
      set_network_input(_network_input);
   else
      validateNetworkInput(_network_input);

   /*
   if (!network_input_set)
   {
      static std::stringstream sout;
      sout.clear();
      sout
         << "Error : BasicNeuralNetFactory::add_layer_external_input() - "
         << "Can't add external basiclayer input - network input not yet specified"
         << std::endl;
      throw std::logic_error (sout.str ());
   }
    */

   OldNetworkLayer& to = *layers[_to];

   size_t to_input_sz = to.add_external_input(network_input, _indexSet);

   layer_external_input_set = true;
}

void BasicNeuralNetFactory::validate()
{
   if (!network_input_set)
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : BasicNeuralNetFactory::validate_network() - "
           << "Can't build network - no network input set_weights." << std::endl;
      throw std::logic_error(sout.str());
   }

   if (layers.size() == 0)
   {
      static std::stringstream sout;
      sout.clear();
      sout << "Error : BasicNeuralNetFactory::validate_network() - "
           << "Can't build network - no network layers specified."
           << std::endl;
      throw std::logic_error(sout.str());
   }

   /*
    * Check that at least one basiclayer receives external network input.
    */
   if (!layer_external_input_set)
   {
      static std::stringstream sout;
      sout.clear();
      sout
         << "Error : BasicNeuralNetFactory::validate_network() - "
         << "No basiclayer external inputs have been set_weights."
         << std::endl;
      throw std::logic_error(sout.str());
   }

   /*
    * Check that there is at least one network basiclayer designated as an network output basiclayer.
    */
   if (outputLayerCount() == 0)
   {
      static std::stringstream sout;
      sout.clear();
      sout
         << "Error : BasicNeuralNetFactory::validate_network() - "
         << "No output layers have been designated."
         << std::endl;
      throw std::logic_error(sout.str());
   }

   /*
    * Check that all network layers are being used. All used layers must have inputs from
    * other layers and/or external network inputs.
    */
   std::set<std::string> bad_layers = checkLayerInputSize();
   if (bad_layers.size() > 0)
   {
      static std::stringstream sout;
      sout.clear();
      sout
         << "Error : BasicNeuralNetFactory::validate_network() - "
         << "basiclayer \"" << bad_layers.begin()->c_str() << "\" has no basiclayer inputs specified."
         << std::endl;
      throw std::logic_error(sout.str());
   }
}

/**
 * Verify that the specified network input matches the previously specified input
 * @param _network_input
 */
void BasicNeuralNetFactory::validateNetworkInput(const Datum& _network_input)
{
   static std::stringstream sout;

   bool good = false;
   if (_network_input.size() != network_input.size())
   {
      sout.clear();
      sout
         << "Error : BasicNeuralNetFactory::validate_network_input() - "
         << "The network input size does not match previously specified network input."
         << std::endl;
      throw std::logic_error(sout.str());
   }

   for (unsigned int i = 0; i < network_input.size(); i++)
      if (_network_input[i].size() != network_input[i].size())
      {
         sout.clear();
         sout
            << "Error : BasicNeuralNetFactory::validate_network_input() - "
            << "The network input at index " << i << " does not match previously specified size."
            << std::endl;
         throw std::logic_error(sout.str());
      }
}

unsigned int BasicNeuralNetFactory::outputLayerCount()
{
   unsigned int count = 0;
   for (auto& layer_name : layer_activation_order)
   {
      if (layers[layer_name]->is_output_layer())
         count++;
   }

   return count;
}

std::set<std::string> BasicNeuralNetFactory::checkLayerInputSize()
{
   std::set<std::string> bad_layers;

   unsigned int count = 0;
   for (auto& layer_name : layer_activation_order)
   {
      if (layers[layer_name]->virtual_input_size() == 0)
         bad_layers.insert(layer_name);
   }

   return bad_layers;
}

/**
 * Return a set containing the names of layers directly through forward
 * connections, feeding activity into the basiclayer, _name.
 *
 * @param _dependencies
 * @param _name
 */
void BasicNeuralNetFactory::getForwardDependencies(std::set<std::string>& _dependencies, const std::string& _name)
{
   OldNetworkLayer& netlayer = *layers[_name];

   const std::vector<OldLayerConnRecord> layer_input_records = netlayer.get_input_connections();

// Add indirect dependencies by recursing on direct dependencies not already in dependency set_weights
   for (auto& record : layer_input_records)
   {
      // Ignore recurrent connections
      if (record.is_recurrent())
         continue;

      /*
       * If this basiclayer is not already in our dependency list then
       * add it and recurse; otherwise we've recursed this path already
       * so do nothing.
       */
      if (_dependencies.find(record.get_input_layer().name())
          == _dependencies.end())
      {
         _dependencies.insert(record.get_input_layer().name());
         getForwardDependencies(_dependencies, record.get_input_layer().name());
      }
   }
}

/**
 * Return a set containing the names of layers through forward or recurrent
 * connections, feeding activity into the basiclayer, _name.
 *
 * @param _dependencies
 * @param _name
 */
void BasicNeuralNetFactory::getAllDependencies(std::set<std::string>& _dependencies, const std::string& _name)
{
   OldNetworkLayer& network_layer = *layers[_name];

   const std::vector<OldLayerConnRecord> layer_input_records = network_layer.get_input_connections();

// Add indirect dependencies by recursing on direct dependencies not already in dependency set_weights
   for (auto& record : layer_input_records)
   {
      /*
       * If this basiclayer is not already in our dependency list then
       * add it and recurse; otherwise we've recursed this path already
       * so do nothing.
       */
      if (_dependencies.find(record.get_input_layer().name())
          == _dependencies.end())
      {
         _dependencies.insert(record.get_input_layer().name());
         getAllDependencies(_dependencies, record.get_input_layer().name());
      }
   }
}

void BasicNeuralNetFactory::updateActivationOrder(void)
{
   layer_activation_order.clear();
   for (auto& item : layers)
   {
      std::string layer_name = item.first;

      /*
       * Find the first (if any) basiclayer already in the ordered activation list
       * that has a feedforward input dependency on the new basiclayer and insert the
       * new basiclayer just in front of it.
       */
      bool inserted = false;
      for (auto it = layer_activation_order.begin(); it != layer_activation_order.end(); ++it)
      {
         auto& ordered_layer_name = *it;

         // Get dependencies for ordered_layer_name
         std::set<std::string> dependencies;
         getForwardDependencies(dependencies, ordered_layer_name);

         // If new layer_name is in the list then break now and insert.
         if (dependencies.find(layer_name) != dependencies.end())
         {
            layer_activation_order.insert(it, layer_name);
            inserted = true;
            break;
         }
      }

      // If it wasn't already inserted, add it to the end now
      if (!inserted)
         layer_activation_order.push_back(layer_name);
   }
}


