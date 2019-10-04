//
// Created by kfedrick on 5/11/19.
//

#include <sstream>
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "LayerWeights.h"

using namespace flexnnet;

/**
 * Protected method to resize bias vector and weight array
 *
 * @param _layer_sz
 * @param _layer_input_sz
 */
void LayerWeights::resize (size_t _layer_sz, size_t _layer_input_sz)
{
   if (_layer_sz > 0 && _layer_input_sz > 0)
      weights.resize (_layer_sz, _layer_input_sz + 1);
}

void LayerWeights::set_weights (const Array2D<double> &_weights)
{
   weights = _weights;
}

/**
 * Adjust layer weights by the specified delta weight array.
 */
void LayerWeights::adjust_weights (const Array2D<double> &_delta)
{
   weights += _delta;
}

std::string LayerWeights::to_json (void) const
{
   std::string json;
   return json;
}