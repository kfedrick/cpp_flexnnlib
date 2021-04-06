//
// Created by kfedrick on 3/12/21.
//

#include "Globals.h"

#include <cxxabi.h>

std::string flexnnet::demangle(const std::string& name)
{
   static char buf[2048];
   size_t size = sizeof(buf);
   int status;

   char* res = abi::__cxa_demangle(name.c_str(), buf, &size, &status);
   buf[sizeof(buf) - 1] = 0;

   return std::string(buf);
}