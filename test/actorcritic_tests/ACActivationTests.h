//
// Created by kfedrick on 5/21/21.
//

#ifndef _ACACTIVATIONTESTS_H_
#define _ACACTIVATIONTESTS_H_

#include <gtest/gtest.h>
#include <CommonTestFixtureFunctions.h>
#include <Reinforcement.h>

class ACActivationTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:

   virtual void
   SetUp()
   {}

   virtual void
   TearDown()
   {}


};

TEST_F(ACActivationTestFixture, ReinfConstructorTest)
{
   flexnnet::Reinforcement<1> r;
}

#endif //_ACACTIVATIONTESTS_H_
