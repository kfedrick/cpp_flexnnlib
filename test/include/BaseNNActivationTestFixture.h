//
// Created by kfedrick on 2/27/21.
//

#ifndef _BASENNACTIVATIONTESTFIXTURE_H_
#define _BASENNACTIVATIONTESTFIXTURE_H_

#include "CommonTestFixtureFunctions.h"

template<typename T>
class BaseNNActivationTestFixture
{
public:
   std::string get_typeid();
};


template<typename T> std::string BaseNNActivationTestFixture<T>::get_typeid()
{
   std::string type_id = typeid(T).name();
   static char buf[2048];
   size_t size = sizeof(buf);
   int status;

   char* res = abi::__cxa_demangle(type_id.c_str(), buf, &size, &status);
   buf[sizeof(buf) - 1] = 0;

   std::string buf_str = buf;
   size_t pos = buf_str.rfind(':') + 1;
   buf_str = buf_str.substr(pos);

   return buf_str;
}

#endif //_BASENNACTIVATIONTESTFIXTURE_H_
