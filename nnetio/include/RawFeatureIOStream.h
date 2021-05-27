//
// Created by kfedrick on 5/23/21.
//

#ifndef FLEX_NEURALNET_RAWFEATUREIOSTREAM_H_
#define FLEX_NEURALNET_RAWFEATUREIOSTREAM_H_

#include <cxxabi.h>
#include "picojson.h"

#include <RawFeature.h>

template<size_t N>
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::RawFeature<N>& _feature)
{
   const std::valarray<double>& v = _feature.get_encoding();

   _ostrm << "[";
   for (size_t vndx = 0; vndx < v.size() - 1; vndx++)
      _ostrm << v[vndx] << ",";
   _ostrm << v[v.size() - 1] << "]";

   return _ostrm;
}

template<size_t N> std::istream& operator>>(std::istream& _istrm, flexnnet::RawFeature<N>& _feature)
{
   picojson::value picoval;

   // Read next json vectorize
   _istrm >> picoval;

   const picojson::array& arr = picoval.get<picojson::array>();

   std::valarray<double> va(arr.size());
   for (int i = 0; i < arr.size(); i++)
      va[i] = arr[i].get<double>();
   _feature.decode(va);

   // Return the remainder of the input stream
   return _istrm;
}

#endif // FLEX_NEURALNET_RAWFEATUREIOSTREAM_H_
