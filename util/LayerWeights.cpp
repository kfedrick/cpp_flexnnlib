//
// Created by kfedrick on 5/11/19.
//

#include <sstream>
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "LayerWeights.h"

using std::string;
using flexnnet::LayerWeights;

LayerWeights::LayerWeights(void) : NamedObject("LayerWeights")
{
}

LayerWeights::LayerWeights(const LayerWeights& _lweights) : NamedObject(_lweights.name())
{
   copy(_lweights);
}

LayerWeights::LayerWeights(const LayerWeights&& _lweights) : NamedObject(_lweights.name())
{
   copy(_lweights);
}

/**
 * Protected method to resize bias vector and weight array
 *
 * @param _layer_sz
 * @param _layer_input_sz
 */
void LayerWeights::resize (size_t _layer_sz, size_t _layer_input_sz)
{
   if (_layer_sz > 0 && _layer_input_sz > 0)
   {
      weights.resize(_layer_sz, _layer_input_sz + 1);
      initial_value.resize(_layer_sz);
   }
}

std::string LayerWeights::to_json (void) const
{
   std::string json;
   return json;
}