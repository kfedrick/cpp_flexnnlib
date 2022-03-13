//
// Created by kfedrick on 5/24/21.
//

#ifndef FLEX_NEURALNET_RAWFEATURESET_H_
#define FLEX_NEURALNET_RAWFEATURESET_H_

#include <FeatureSetImpl.h>
#include <RawFeature.h>

namespace flexnnet
{
   template<size_t ...N> class RawFeatureSet : public FeatureSetImpl<std::tuple<RawFeature<N>...>>
   {
   public:
      RawFeatureSet() : FeatureSetImpl<std::tuple<RawFeature<N>...>>()
      {};

      RawFeatureSet(const RawFeatureSet& _fs) : FeatureSetImpl<std::tuple<RawFeature<N>...>>(_fs)
      {};

      RawFeatureSet(const std::array<std::string, sizeof...(N)>& _names) : FeatureSetImpl<std::tuple<
         RawFeature<N>...>>(_names)
      {};
   };

} // end namespace flexnnet

#endif // FLEX_NEURALNET_RAWFEATURESET_H_
