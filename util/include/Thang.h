//
// Created by kfedrick on 5/17/21.
//

#ifndef FLEX_NEURALNET_THANG_H_
#define FLEX_NEURALNET_THANG_H_

#include <NamedObject.h>

namespace flexnnet
{
   class Thang : public NamedObject
   {
   public:
      Thang() : NamedObject("Thang") {}
      Thang(const std::string& _id) : NamedObject(_id) {}
   };
}

#endif //_THANG_H_
