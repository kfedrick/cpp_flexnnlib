/*
 * ArrayInitializer.cpp
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#include "ArrayInitializer.h"

#include <stdio.h>

namespace flex_neuralnet
{


ArrayInitializer::~ArrayInitializer()
{
}

void ArrayInitializer::operator()(Array<double>& arr) const
{
   printf ("ArrayInitializer::operator()\n");
}

ArrayInitializer* ArrayInitializer::clone() const
{
   return new ArrayInitializer(*this);
}

} /* namespace flex_neuralnet */
