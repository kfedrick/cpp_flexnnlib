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
void LayerWeights::resize (unsigned int _layer_sz, unsigned int _layer_input_sz)
{
   if (_layer_sz > 0 && _layer_input_sz > 0)
      weights.resize (_layer_sz, _layer_input_sz+1);
}

void LayerWeights::set_weights (const Array<double> &_weights)
{
   weights = _weights;
}

/**
 * Adjust layer weights by the specified delta weight array.
 */
void LayerWeights::adjust_weights (const Array<double> &_delta)
{
   for (unsigned int row = 0; row < _delta.rowDim (); row++)
      for (unsigned int col = 0; col < _delta.colDim (); col++)
         weights.at (row, col) += _delta.at (row, col);
}

std::string LayerWeights::to_json (void) const
{
   std::string json;
   return json;
}