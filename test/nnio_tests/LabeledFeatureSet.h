//
// Created by kfedrick on 5/23/21.
//

#ifndef _MYFEATURESET_H_
#define _MYFEATURESET_H_

#include <FeatureSet.h>
#include <FixedSizeFeature.h>

typedef flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>,flexnnet::RawFeature<1>>> MyFeatureSetType;

class LabeledFeatureSet : public MyFeatureSetType
{
public:
   static std::array<std::string, 2> my_feature_names;

public:
   LabeledFeatureSet();
};

std::array<std::string, 2> LabeledFeatureSet::my_feature_names = {"Feature0", "Feature1"};

LabeledFeatureSet::LabeledFeatureSet() : MyFeatureSetType(my_feature_names)
{
   //StateFs::feature_names = my_feature_names;
}


#endif // FLEX_NEURALNET_MYFEATURESET_H_
