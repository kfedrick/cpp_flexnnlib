//
// Created by kfedrick on 5/23/21.
//

#ifndef FLEX_NEURALNET_FEATURESETIOSTREAM_H_
#define FLEX_NEURALNET_FEATURESETIOSTREAM_H_

#include <cxxabi.h>
#include "picojson.h"

#include <FeatureSet.h>

/*
 * Forward function declarations
 */
template<size_t I, typename ...Fs>
typename std::enable_if<I == sizeof...(Fs), std::ostream&>::type
ostream_helper(std::ostream& _ostrm, const std::array<std::string, sizeof...(Fs)> _labels, const std::tuple<Fs...>& _features);

template<size_t I=0, typename ...Fs>
typename std::enable_if<I < sizeof...(Fs), std::ostream&>::type
ostream_helper(std::ostream& _ostrm, const std::array<std::string, sizeof...(Fs)> _labels, const std::tuple<Fs...>& _features);

template<size_t I, typename ...Fs>
typename std::enable_if<I == sizeof...(Fs), std::istream&>::type
istream_helper(std::istream& _istrm, const std::array<std::string, sizeof...(Fs)> _labels, const picojson::object& _fmap, std::tuple<Fs...>& _features);

template<size_t I=0, typename ...Fs>
typename std::enable_if<I < sizeof...(Fs), std::istream&>::type
istream_helper(std::istream& _istrm, const std::array<std::string, sizeof...(Fs)> _labels, const picojson::object& _fmap, std::tuple<Fs...>& _features);

/*
 * input/output stream operators
 */

template<typename ...Fs>
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::FeatureSet<std::tuple<Fs...>>& _featureset)
{
   std::array<std::string,sizeof...(Fs)> labels = _featureset.get_feature_names();
   const std::tuple<Fs...>& features = _featureset.get_features();

   _ostrm << "{\n";
   ostream_helper<0, Fs...>(_ostrm, labels, features);
   _ostrm << "\n}";

   return _ostrm;
}

template<typename ...Fs>
std::istream& operator>>(std::istream& _istrm, flexnnet::FeatureSet<std::tuple<Fs...>>& _featureset)
{
   std::array<std::string,sizeof...(Fs)> labels = _featureset.get_feature_names();
   std::tuple<Fs...>& features = _featureset.get_features();

   picojson::value picoval;

   // Read next json vectorize
   _istrm >> picoval;

   const picojson::object& fmap = picoval.get<picojson::object>();
   istream_helper<0>(_istrm, labels, fmap, features);

   // Return the remainder of the input stream
   return _istrm;
}

template<size_t I, typename ...Fs>
typename std::enable_if<I == sizeof...(Fs), std::ostream&>::type
ostream_helper(std::ostream& _ostrm, const std::array<std::string, sizeof...(Fs)> _labels, const std::tuple<Fs...>& _features)
{
   return _ostrm;
}

template<size_t I, typename ...Fs>
typename std::enable_if<I < sizeof...(Fs), std::ostream&>::type
ostream_helper(std::ostream& _ostrm, const std::array<std::string, sizeof...(Fs)> _labels, const std::tuple<Fs...>& _features)
{
   if (I > 0)
      _ostrm << ",\n";

   _ostrm << "  \"" << _labels[I] << "\":";
   _ostrm << std::get<I>(_features);

   ostream_helper<I + 1>(_ostrm, _labels, _features);

   return _ostrm;
}

template<size_t I, typename ...Fs>
typename std::enable_if<I == sizeof...(Fs), std::istream&>::type
istream_helper(std::istream& _istrm, const std::array<std::string, sizeof...(Fs)> _labels, const picojson::object& _fmap, std::tuple<Fs...>& _features)
{
   return _istrm;
}

template<size_t I, typename ...Fs>
typename std::enable_if<I < sizeof...(Fs), std::istream&>::type
istream_helper(std::istream& _istrm, const std::array<std::string, sizeof...(Fs)> _labels, const picojson::object& _fmap, std::tuple<Fs...>& _features)
{
   const picojson::value& v = _fmap.at(_labels[I]);

   std::stringstream ss;
   ss.str(v.serialize());
   ss >> std::get<I>(_features);

   istream_helper<I + 1>(_istrm, _labels, _fmap, _features);

   return _istrm;
}
#endif // FLEX_NEURALNET_FEATURESETIOSTREAM_H_
