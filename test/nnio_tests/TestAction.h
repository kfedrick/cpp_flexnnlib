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

class TestAction : public flexnnet::FeatureSetImpl<TestActionFeature>
{
public:
   TestAction();
   TestAction(const FeatureSetImpl<TestActionFeature>& _ta);
   TestAction& operator=(const FeatureSetImpl<TestActionFeature>& _ta);
};

TestAction::TestAction() : FeatureSetImpl<TestActionFeature>()
{
}

TestAction::TestAction(const FeatureSetImpl<TestActionFeature>& _ta) : FeatureSetImpl<TestActionFeature>(_ta)
{
}

TestAction& TestAction::operator=(const FeatureSetImpl<TestActionFeature>& _ta)
{
}



#endif //_TESTACTION_H_
