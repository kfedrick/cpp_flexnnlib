//
// Created by kfedrick on 5/21/21.
//

#ifndef _TESTACTION_H_
#define _TESTACTION_H_

#include <ActionOutput.h>
#include <NetworkOutput.h>
#include <NetworkLayer.h>
#include <FeatureSetImpl.h>
#include <RawFeature.h>
#include "TestActionFeature.h"

class TestAction : public flexnnet::FeatureSetImpl<std::tuple<TestActionFeature>>
{
public:
   TestAction();
   TestAction(const FeatureSetImpl<std::tuple<TestActionFeature>>& _ta);
   TestAction& operator=(const FeatureSetImpl<std::tuple<TestActionFeature>>& _ta);
};

TestAction::TestAction() : FeatureSetImpl<std::tuple<TestActionFeature>>()
{
}

TestAction::TestAction(const FeatureSetImpl<std::tuple<TestActionFeature>>& _ta) : FeatureSetImpl<std::tuple<TestActionFeature>>(_ta)
{
   //std::cout << "TestAction::copy constructor\n";
   //std::cout << "value<0>[0] " << std::get<0>(this->get_features()).get_encoding()[0] << "\n";

   *this = _ta;
}

TestAction& TestAction::operator=(const FeatureSetImpl<std::tuple<TestActionFeature>>& _ta)
{
   //std::cout << "TestAction::operator=\n";
   //std::cout << "value<0>[0] " << std::get<0>(this->get_features()).get_encoding()[0] << "\n";
   //std::cout << "ta.value<0>[0] " << std::get<0>(_ta.get_features()).get_encoding()[0] << "\n";

   FeatureSetImpl<std::tuple<TestActionFeature>>::operator=(_ta);
   //std::cout << "value<0>[0] after " << std::get<0>(this->get_features()).get_encoding()[0] << "\n";

   return *this;
}



#endif //_TESTACTION_H_
