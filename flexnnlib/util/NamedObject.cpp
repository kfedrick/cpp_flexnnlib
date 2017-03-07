/*
 * NamedObject.cpp
 *
 *  Created on: Feb 5, 2014
 *      Author: kfedrick
 */

#include "NamedObject.h"

namespace flex_neuralnet
{

NamedObject::NamedObject(const string& _name)
{
   myname = _name;
}

NamedObject::NamedObject(const char* _name)
{
   myname = _name;
}

const string& NamedObject::name() const
{
   return myname;
}

void NamedObject::copy(const NamedObject& _namedObj)
{
   myname = _namedObj.myname;
}

} /* namespace flex_neuralnet */
