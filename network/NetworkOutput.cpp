//
// Created by kfedrick on 2/22/21.
//

#include "NetworkOutput.h"

using flexnnet::NetworkOutput;

NetworkOutput::NetworkOutput() : NetworkLayer()
{
   layer_name = "__network_output";
}

NetworkOutput::~NetworkOutput() {}

const std::valarray<double>& NetworkOutput::activate(const ValarrMap& _externin)
{
   marshal_inputs(_externin);
   concat_inputs(_externin);
   return virtual_input_vector_const_ref;
}

