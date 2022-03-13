//
// Created by kfedrick on 3/12/22.
//

#ifndef FLEX_NEURALNET_FEATURESET_H_
#define FLEX_NEURALNET_FEATURESET_H_

#include <cstdlib>
#include <vector>
#include <string>
#include <Feature.h>

namespace flexnnet
{
   class FeatureSet
   {
   public:
      virtual size_t size() const =0;
      virtual const std::vector<std::string>& get_feature_namesv() const =0;
      virtual size_t get_feature_index(const std::string& _id) const =0;

      virtual Feature& operator[](size_t _ndx) =0;
      virtual const Feature& operator[](size_t _ndx) const =0;
      virtual Feature& at(const std::string& _id) =0;
      virtual const Feature& at(const std::string& _id) const =0;
      virtual void decode(const std::vector<std::valarray<double>>& _encodings) =0;
   };
}

#endif // FLEX_NEURALNET_FEATURESET_H_
