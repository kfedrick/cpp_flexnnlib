//
// Created by kfedrick on 3/12/21.
//

#include "JSONStream.h"

using flexnnet::JSONStream;



std::string JSONStream::tojson(const std::valarray<double>& _valarr, size_t _indent)
{
   std::ostringstream ss;
   std::string spaces(_indent, ' ');

   ss << spaces << "[";

   unsigned int sz = _valarr.size();
   for (unsigned int ndx=0; ndx < sz-1; ndx++)
      ss << spaces << "  " << _valarr[ndx] << ",";
   ss << spaces << "  " << _valarr[sz-1];

   ss << spaces << "]";

   return ss.str();
}

std::string JSONStream::tojson(const std::string& _val, size_t _indent)
{
   std::ostringstream ss;

   ss << "\"" << _val.c_str() << "\"";
   return ss.str();
}

template <typename _Typ>
std::string JSONStream::tojson(const std::vector<_Typ>& _vec, size_t _indent)
{
   std::ostringstream ss;
   std::string spaces(_indent, ' ');

   ss << spaces << "[";

   unsigned int sz = _vec.size();
   for (unsigned int ndx=0; ndx < sz-1; ndx++)
      ss << spaces << "  " << _vec[ndx] << ",";
   ss << spaces << "  " << _vec[sz-1];

   ss << spaces << "]";

   return ss.str();
}
