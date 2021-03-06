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

LayerWeights::LayerWeights(void)
{
}

LayerWeights::LayerWeights(const std::vector<std::vector<double>>& _lweights)
{
   set(_lweights);
}

LayerWeights::LayerWeights(const Array2D<double>& _lweights)
{
   Array2D<double>::Dimensions dim = _lweights.size();
   resize(dim.rows, dim.cols-1);
   set(_lweights);
}

LayerWeights::LayerWeights(const LayerWeights& _lweights)
{
   copy(_lweights);
}

LayerWeights::LayerWeights(const LayerWeights&& _lweights)
{
   copy(_lweights);
}

/**
 * Protected method to resize bias vector and weight array
 *
 * @param _layer_sz
 * @param _layer_input_sz
 */
void LayerWeights::resize(size_t _layer_sz, size_t _layer_input_sz)
{
   if (_layer_sz > 0 && _layer_input_sz > 0)
   {
      weights.resize(_layer_sz, _layer_input_sz + 1);
      initial_layer_value.resize(_layer_sz);
   }
}

std::string LayerWeights::to_json(void) const
{
   std::string json;
   return json;
}