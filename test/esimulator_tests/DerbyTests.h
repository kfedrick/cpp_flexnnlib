//
// Created by kfedrick on 5/9/21.
//

#ifndef _DERBYTESTS_H_
#define _DERBYTESTS_H_

#include <gtest/gtest.h>
#include <CommonTestFixtureFunctions.h>
#include "DerbySim.h"
#include "SteeringAction.h"
#include "Reinforcement.h"

class DerbySimTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   virtual void
   SetUp()
   {}
   virtual void
   TearDown()
   {}
};

TEST_F(DerbySimTestFixture, Constructor)
{
   std::cout << "Derby Constructor Test\n";

   DerbySim derbysim;

}

TEST_F(DerbySimTestFixture, Reset)
{
   std::cout << "Derby Reset Test\n";

   DerbySim derbysim;
   flexnnet::RawFeatureSet<12> state = derbysim.reset();
   flexnnet::ValarrMap vmap = state.value_map();

   std::cout << "state size = " << state.size() << "\n";
   std::cout << "state vector size = " << state.size() << "\n";
   std::cout << this->prettyPrintVector("state(vectorize)", vmap["F0"]) << "\n";
}

TEST_F(DerbySimTestFixture, Left)
{
   std::cout << "Derby Left Test\n";

   DerbySim derbysim;

   derbysim.reset();
   const flexnnet::RawFeatureSet<12>& state = derbysim.state();
   flexnnet::ValarrMap vmap = state.value_map();

   std::cout << this->prettyPrintVector("start state", vmap["F0"]) << "\n";

   derbysim.next(ActionEnum::Left);
   const flexnnet::RawFeatureSet<12>& newstate = derbysim.state();
   flexnnet::ValarrMap newvmap = newstate.value_map();

   std::cout << this->prettyPrintVector("next(Left) state", newvmap["F0"]) << "\n";
}

TEST_F(DerbySimTestFixture, Right)
{
   std::cout << "Derby Right Test\n";

   DerbySim derbysim;

   const flexnnet::RawFeatureSet<12>& state = derbysim.state();
   flexnnet::ValarrMap vmap = state.value_map();

   std::cout << this->prettyPrintVector("start state", vmap["F0"]) << "\n";

   derbysim.next(ActionEnum::Right);
   const flexnnet::RawFeatureSet<12>& newstate = derbysim.state();
   flexnnet::ValarrMap newvmap = newstate.value_map();

   std::cout << this->prettyPrintVector("next(Right) state", newvmap["F0"]) << "\n";
}

TEST_F(DerbySimTestFixture, NotTerminal)
{
   std::cout << "Derby NotTerminal Test\n";

   DerbySim derbysim;

   const flexnnet::RawFeatureSet<12>& state = derbysim.state();
   flexnnet::ValarrMap vmap = state.value_map();

   std::cout << this->prettyPrintVector("start state", vmap["F0"]) << "\n";

   bool terminal = derbysim.is_terminal();
   EXPECT_FALSE(terminal);
}

TEST_F(DerbySimTestFixture, Reinforcement)
{
   std::cout << "Derby NetworkReinforcement Test\n";

   DerbySim derbysim;

   const flexnnet::RawFeatureSet<12>& state = derbysim.state();
   flexnnet::ValarrMap vmap = state.value_map();

   std::cout << this->prettyPrintVector("start state", vmap["F0"]) << "\n";

   derbysim.get_reinforcement();
   const flexnnet::Reinforcement<1>& r = derbysim.get_reinforcement();

   std::cout << "EnvironReinforcement values " << r.size() << "\n" << std::flush;
   for (int i=0; i<r.size(); i++)
      std::cout << r[i] << "\n";
}

TEST_F(DerbySimTestFixture, ReinforcementConstructor)
{
   flexnnet::Reinforcement<2> r;

   std::cout << "EnvironReinforcement values " << r.size() << "\n" << std::flush;
   for (int i=0; i<r.size(); i++)
      std::cout << i << " " << r.get_feature_names()[i] << "\n";
}

TEST_F(DerbySimTestFixture, SteeringActionConstructor)
{
   std::cout << "Derby SteeringAction Test\n";

   SteeringAction steer;
}

#endif //_DERBYTESTS_H_
