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

std::shared_ptr<flexnnet::BasicLayer>& NetworkOutput::layer()
{
   static std::stringstream sout;
   sout << "Error : NetworkOutput::layer() - Invalid operation on object of this type.\n";
   throw std::logic_error(sout.str());
}

const std::valarray<double>& NetworkOutput::activate(const NNetIO_Typ& _externin)
{
   marshal_inputs(_externin);
   return virtual_input_vector;
}