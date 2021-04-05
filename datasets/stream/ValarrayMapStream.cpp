//
// Created by kfedrick on 3/14/21.
//


#include <ValarrayMap.h>
#include "picojson.h"

using flexnnet::ValarrayMap;
using flexnnet::ValarrMap;

std::ostream&
operator<<(std::ostream& _ostrm, const ValarrayMap& _vmap)
{
   const ValarrMap& vmap = _vmap.value_map();

   _ostrm << "{\n";

   // Iterate over valarray map
   bool first = true;
   for (auto it : vmap)
   {
      if (!first)
         _ostrm << ",";

      const std::string& key = it.first;
      const std::valarray<double>& v = it.second;

      _ostrm << "   " << "\"" << key << "\"" << ":{";
      for (size_t vndx=0; vndx<v.size()-1; vndx++)
         _ostrm << v[vndx] << ",";
      _ostrm << v[v.size()-1] << "}";

      first = false;
   }
   _ostrm << "\n    }";

   return _ostrm;
}

std::istream&
operator>>(std::istream& _istrm, ValarrayMap& _coord)
{
   picojson::value picoval;

   // Read next json value
   _istrm >> picoval;


   // Return the remainder of the input stream
   return _istrm;
}

