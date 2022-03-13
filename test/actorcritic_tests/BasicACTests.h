//
// Created by kfedrick on 5/17/21.
//

#ifndef _BASICACTESTS_H_
#define _BASICACTESTS_H_

#include <gtest/gtest.h>
#include <CommonTestFixtureFunctions.h>

#include <BaseActorCriticNetwork.h>
#include <NetworkLayerImpl.h>
#include <BaseNeuralNet.h>
#include <NeuralNet.h>
#include <RawFeature.h>
#include <FeatureSetImpl.h>
#include <Reinforcement.h>

#include <TanSig.h>
#include "TestAction.h"

#include "State.h"
#include "TestActionFeature.h"
#include <RawFeatureSet.h>

using flexnnet::FeatureSetImpl;
using flexnnet::RawFeature;
using flexnnet::RawFeatureSet;

class BasicACTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   virtual void
   SetUp()
   {}
   virtual void
   TearDown()
   {}
};


TEST_F(BasicACTestFixture, CriticNetConstructor)
{
   std::cout << "Critic network Constructor Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol_ptr =
                                                std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "output", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 1);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<RawFeatureSet<3>, flexnnet::Reinforcement<1>> critic(basecritic);
}

TEST_F(BasicACTestFixture, SingleCriticActivate)
{
   std::cout << "Single Critic Activate Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "F0", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("F0", 3);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<RawFeatureSet<3>, flexnnet::Reinforcement<1>> critic(basecritic);
   critic.initialize_weights();

   RawFeatureSet<3> f;
   f.decode({{0, 1, 2}});

   flexnnet::Reinforcement<1> nnout = critic.activate(f);
   std::cout << "vectorized size " << std::get<0>(nnout.get_features()).size() << "\n" << std::flush;
   //std::cout << prettyPrintVector("nnout", nnout.vectorize());
}

TEST_F(BasicACTestFixture, MultiCriticAccessField)
{
   std::cout << "Multi Critic Access by field Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol1_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(2, "F0", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol1_ptr->add_external_input_field("F0", 3);

   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol1_ptr->name()] = ol1_ptr;
   topo.network_output_layers.push_back(ol1_ptr);
   topo.ordered_layers.push_back(ol1_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<RawFeatureSet<3>, flexnnet::Reinforcement<2>> critic(basecritic);
   critic.initialize_weights();

   RawFeatureSet<3> f;
   f.decode({{0, 1, 2}});
   critic.activate(f);

   flexnnet::Reinforcement<2> nnout = critic.activate(f);

   std::array<std::string,1> fields = nnout.get_feature_names();
   for (auto a_field : fields)
   {
      //std::cout << a_field << " " << nnout.at("F0") << "\n";
      std::cout << a_field << " " << std::get<0>(nnout.get_features()).value()[0] << "\n";
   }
}

TEST_F(BasicACTestFixture, MultiCriticAccessIndex)
{
   std::cout << "Multi Critic Access by Index Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol1_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "F0", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol1_ptr->add_external_input_field("F0", 3);

   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol1_ptr->name()] = ol1_ptr;
   topo.network_output_layers.push_back(ol1_ptr);
   topo.ordered_layers.push_back(ol1_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<RawFeatureSet<3>, flexnnet::Reinforcement<2>> critic(basecritic);
   critic.initialize_weights();

   RawFeatureSet<3> f;
   f.decode({{0, 1, 2}});
   critic.activate(f);

   flexnnet::Reinforcement<2> nnout = critic.activate(f);

   std::array<std::string,1> fields = nnout.get_feature_names();
   for (int ndx=0; ndx<fields.size(); ndx++)
   {
      std::cout << ndx << " " << fields[ndx] << " " << std::get<0>(nnout.get_features()).value()[ndx] << "\n";
   }
}

TEST_F(BasicACTestFixture, MultiCriticAccessIndexAt)
{
   std::cout << "Multi Critic Access by Index at Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol1_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "F0", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol1_ptr->add_external_input_field("F0", 3);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol1_ptr->name()] = ol1_ptr;
   topo.network_output_layers.push_back(ol1_ptr);
   topo.ordered_layers.push_back(ol1_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<RawFeatureSet<3>, flexnnet::Reinforcement<2>> critic(basecritic);
   critic.initialize_weights();

   RawFeatureSet<3> f;
   f.decode({{0, 1, 2}});

   flexnnet::Reinforcement<2> nnout = critic.activate(f);

   EXPECT_EQ(nnout.size(), 1) << "Reinforcement size not correct\n";

   std::array<std::string,1> fields = nnout.get_feature_names();
   for (int ndx=0; ndx<fields.size(); ndx++)
   {
      std::cout << ndx << " " << fields[ndx] << " " << std::get<0>(nnout.get_features()).value()[ndx] << "\n";
   }
}
template<typename T1, typename T2>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<T1>(),
                                            std::declval<T2>()));

TEST_F(BasicACTestFixture, TuplecatTest)
{
   std::cout << "tuple_cat Test\n" << std::flush;

   typedef std::tuple<int, long, char> T1;
   typedef std::tuple<float, double> T2;

   T1 t1(1,666,'a');
   T2 t2(3.14159, 2.17);

   auto t3 = std::tuple_cat(t1, t2);

   std::cout << std::get<0>(t3) << " " << std::get<3>(t3) << "\n";

   tuple_cat_t<T1,T2> t4;
   t4 = t3;

   std::cout << std::get<0>(t4) << " " << std::get<3>(t4) << "\n";
}

class TS1 : public std::tuple<int, long, char> {};
class TS2 : public std::tuple<float, double> {};

template<typename T1, typename T2>
using tuple_cat2_t = decltype(std::tuple_cat(std::declval<T1>().get_features(),
                                            std::declval<T2>().get_features()));

TEST_F(BasicACTestFixture, TupleClassTest)
{
   std::cout << "tuple_cat featureset Test\n" << std::flush;

   RawFeatureSet<2> fs1;
   RawFeatureSet<1,1> fs2;

   fs1.decode({{3.14159,2.17}});
   fs2.decode({{9.5},{666.0}});

   typedef tuple_cat2_t<RawFeatureSet<2>,RawFeatureSet<1,1>> TCAT;
   TCAT tcat;
   tcat = std::tuple_cat(fs1.get_features(), fs2.get_features());

   std::cout << "sizeof concat tuple " << std::tuple_size<TCAT>{} << "\n";
   std::cout << std::get<0>(tcat).get_encoding()[0] << " "
             << std::get<0>(tcat).get_encoding()[1] << " "
             << std::get<1>(tcat).get_encoding()[0] << " "
             << std::get<2>(tcat).get_encoding()[0] << "\n";

}

#endif //_BASICACTESTS_H_
