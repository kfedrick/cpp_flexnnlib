//
// Created by kfedrick on 5/17/21.
//

#ifndef FLEX_NEURALNET_THING_H_
#define FLEX_NEURALNET_THING_H_

#include <NamedObject.h>

namespace flexnnet
{
   class Thing : public NamedObject
   {
   public:
      Thing() : NamedObject("Thing") {}
      Thing(const std::string& _id) : NamedObject(_id) {}
   };
}

#endif //_THING_H_
