//
// Created by kfedrick on 5/24/21.
//

#ifndef FLEX_NEURALNET_RAWFEATURESET_H_
#define FLEX_NEURALNET_RAWFEATURESET_H_

#include <FeatureSet.h>
#include <RawFeature.h>

namespace flexnnet
{
   template<size_t ...N>
class RawFeatureSet : public FeatureSet<std::tuple<RawFeature<N>...>>
   {
   public:
      RawFeatureSet() : FeatureSet<std::tuple<RawFeature<N>...>>()
      {};

      RawFeatureSet(const std::array<std::string, sizeof...(N)>& _names) : FeatureSet<std::tuple<RawFeature<N>...>>(_names)
      {};
   };

} // end namespace flexnnet

#endif // FLEX_NEURALNET_RAWFEATURESET_H_
