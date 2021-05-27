//
// Created by kfedrick on 5/21/21.
//

#ifndef _TESTACTION_H_
#define _TESTACTION_H_

#include <ActionOutput.h>
#include <NetworkOutput.h>
#include <NetworkLayer.h>
#include <FeatureSet.h>
#include <RawFeature.h>
#include "TestActionFeature.h"

class TestAction : public flexnnet::FeatureSet<TestActionFeature>
{
public:
   TestAction();
   TestAction(const FeatureSet<TestActionFeature>& _ta);
   TestAction& operator=(const FeatureSet<TestActionFeature>& _ta);
};

TestAction::TestAction() : FeatureSet<TestActionFeature>()
{
}

TestAction::TestAction(const FeatureSet<TestActionFeature>& _ta) : FeatureSet<TestActionFeature>(_ta)
{
}

TestAction& TestAction::operator=(const FeatureSet<TestActionFeature>& _ta)
{
}



#endif //_TESTACTION_H_
