/*
 * NamedObject.h
 *
 *  Created on: Feb 5, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_NAMEDOBJECT_H_
#define FLEX_NEURALNET_NAMEDOBJECT_H_

#include <string>

using namespace std;
namespace flex_neuralnet
{

class NamedObject
{
public:

   NamedObject(const char* _name  = "NamedObject");
   NamedObject(const string& _name);

   const string& name() const;

protected:
   void copy(const NamedObject& _namedObj);

private:
   string myname;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_NAMEDOBJECT_H_ */
