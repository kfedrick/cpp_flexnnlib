//
// Created by kfedrick on 5/22/21.
//

#ifndef FLEX_NEURALNET_FEATURESETTESTS_H_
#define FLEX_NEURALNET_FEATURESETTESTS_H_

#include <gtest/gtest.h>
#include <CommonTestFixtureFunctions.h>

#include <FeatureSet.h>
#include <FixedSizeFeature.h>
#include <FeatureDecorator.h>
#include <NetworkLayerImpl.h>
#include <TanSig.h>
#include <BaseNeuralNet.h>
#include <NNFeatureSet.h>
#include <NeuralNet.h>
#include <RawFeature.h>
#include "LabeledFeatureSet.h"
#include "TestActionFeature.h"
#include <RawFeatureIOStream.h>
#include <FeatureSetIOStream.h>
#include <RawFeatureSet.h>
#include <Reinforcement.h>

class FeatureSetTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   virtual void
   SetUp()
   {}
   virtual void
   TearDown()
   {}
};

using flexnnet::RawFeature;

TEST_F(FeatureSetTestFixture, Constructor)
{
   std::cout << "\n--- FeatureSet Constructor Test\n" << std::flush;

   flexnnet::FeatureSet<std::tuple<RawFeature<3>>> featureset;

   std::cout << "feature size " << std::get<0>(featureset.get_features()).size() << "\n";
   std::cout << prettyPrintVector("raw feature", std::get<0>(featureset.get_features()).get_encoding());
}

TEST_F(FeatureSetTestFixture, MultiFeatureConstructor)
{
   std::cout << "\n--- FeatureSet MultiFeatureConstructor Test\n" << std::flush;

   flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>, flexnnet::RawFeature<1>>> featureset;

   std::cout << "feature 1 size " << std::get<0>(featureset.get_features()).size() << "\n";
   std::cout << "feature 2 size " << std::get<1>(featureset.get_features()).size() << "\n";
}

TEST_F(FeatureSetTestFixture, Decode)
{
   std::cout << "\n--- FeatureSet Decode Test\n" << std::flush;

   flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>>> featureset;
   featureset.decode({{3.14159, 2.17, 666}});

   std::cout << "feature size " << std::get<0>(featureset.get_features()).size() << "\n";
   std::cout << prettyPrintVector("raw feature", std::get<0>(featureset.get_features()).get_encoding());
}

TEST_F(FeatureSetTestFixture, MultiFeatureDecode)
{
   std::cout << "\n--- FeatureSet MultiFeatureDecode Test\n" << std::flush;

   flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>, flexnnet::RawFeature<1>>> featureset;
   featureset.decode({{3.14159, 2.17, 666}, {9.5}});

   std::cout << prettyPrintVector("raw feature 1", std::get<0>(featureset.get_features()).get_encoding());
   std::cout << prettyPrintVector("raw feature 2", std::get<1>(featureset.get_features()).get_encoding());
}

TEST_F(FeatureSetTestFixture, AssignRawFeatureTest)
{
   std::cout << "\n--- FeatureSet Assign Raw Feature Test\n" << std::flush;

   flexnnet::RawFeature<3> f;
   f.decode({3.14159, 2.17, 666});

   std::cout << prettyPrintVector("raw feature", f.get_encoding());

   flexnnet::RawFeature<3> newf;
   newf = f;
   EXPECT_EQ(f.size(), newf.size()) << "feature should have same size";
   EXPECT_PRED3(vector_double_near, f.get_encoding(), newf.get_encoding(), 0.01) << "ruh roh";

   std::cout << prettyPrintVector("new raw feature", newf.get_encoding());
}

TEST_F(FeatureSetTestFixture, AssignFSofRawTest)
{
   std::cout << "\n--- FeatureSet Assign FS of Raw Test\n" << std::flush;

   flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>, flexnnet::RawFeature<1>>> fs;
   fs.decode({{3.14159, 2.17, 666}, {9.5}});

   std::cout << prettyPrintVector("raw feature 1", std::get<0>(fs.get_features()).get_encoding());
   std::cout << prettyPrintVector("raw feature 2", std::get<1>(fs.get_features()).get_encoding());

   flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>, flexnnet::RawFeature<1>>> newfs;
   newfs = fs;
   EXPECT_EQ(fs.size(), newfs.size()) << "feature set shoud have same size";
   EXPECT_PRED3(vector_double_near, std::get<0>(fs.get_features()).get_encoding(), std::get<0>(newfs.get_features()).get_encoding(), 0.01) << "ruh roh";
   EXPECT_PRED3(vector_double_near, std::get<1>(fs.get_features()).get_encoding(), std::get<1>(newfs.get_features()).get_encoding(), 0.01) << "ruh roh";

   std::cout << prettyPrintVector("new raw feature 1", std::get<0>(newfs.get_features()).get_encoding());
   std::cout << prettyPrintVector("new raw feature 2", std::get<1>(newfs.get_features()).get_encoding());
}

TEST_F(FeatureSetTestFixture, DecoratorTest)
{
   std::cout << "\n--- FeatureSet Decorator Test\n" << std::flush;
   flexnnet::FeatureSet<std::tuple<flexnnet::FeatureDecorator<flexnnet::FixedSizeFeature<3>>>> featureset;

   std::get<0>(featureset.get_features()).decode({3.14159, 2.17, 666});
   std::get<0>(featureset.get_features()).doit();
   std::cout << "decorator name 1 " << std::get<0>(featureset.get_features()).ids[0] << "\n";
   std::cout << "decorator name 2 " << std::get<0>(featureset.get_features()).ids[1] << "\n";
}

TEST_F(FeatureSetTestFixture, DerivedFeatureAssignTest)
{
   std::cout << "\n--- FeatureSet Derived Feature Assignment Test\n" << std::flush;

   TestActionFeature taf;
   taf.decode({1});

   flexnnet::RawFeature<1> f;
   f.decode({3.14159});

   std::cout << prettyPrintVector("deriv feature", taf.get_encoding());
   std::cout << prettyPrintVector("raw feature", f.get_encoding());

   taf = f;
   std::cout << prettyPrintVector("assigned deriv feature", taf.get_encoding());
   EXPECT_PRED3(vector_double_near, taf.get_encoding(), f.get_encoding(), 0.01) << "ruh roh";

}

TEST_F(FeatureSetTestFixture, TestNNFeatureSet)
{
   std::cout << "\n--- FeatureSet NNFeatureSet Test\n" << std::flush;

   // Create NN feature set
   flexnnet::ValueMapFeatureSet<flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>, flexnnet::RawFeature<1>>>> in;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol_ptr = std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "output", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("F0", 3);
   ol_ptr->add_external_input_field("F1", 1);

   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::BaseNeuralNet basennet(topo);

   flexnnet::NeuralNet<flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>, flexnnet::RawFeature<1>>>,
                       flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<1>>>> nnet(basennet);

   in.decode({{3.1419, 2.17, 9.5}, {666}});
   nnet.activate(in);
}

TEST_F(FeatureSetTestFixture, DerivedFeatureSet)
{
   std::cout << "\n--- FeatureSet LabeledFeatureSet Test\n" << std::flush;

   // Create NN feature set
   flexnnet::ValueMapFeatureSet<LabeledFeatureSet> in;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol_ptr = std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "output", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("Feature0", 3);
   ol_ptr->add_external_input_field("Feature1", 1);

   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::BaseNeuralNet basennet(topo);

   flexnnet::NeuralNet<LabeledFeatureSet,
                       flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<1>>>> nnet(basennet);

   in.decode({{3.1419, 2.17, 9.5}, {666}});
   nnet.activate(in);
}

TEST_F(FeatureSetTestFixture, RawFeatureOStream)
{
   std::cout << "\n--- RawFeature ostream Test\n" << std::flush;

   flexnnet::RawFeature<3> f;
   std::valarray<double> v({3.14159,2.17,9.5});
   f.decode(v);

   std::cout << f << "\n";
}

TEST_F(FeatureSetTestFixture, RawFeatureIStream)
{
   std::cout << "\n--- RawFeature istream Test\n" << std::flush;

   flexnnet::RawFeature<3> f;

   std::string objjson = "[3.14159,2.17,9.5]";
   std::stringstream ss;

   ss.str(objjson);
   ss >> f;

   std::cout << prettyPrintVector("raw feature 1", f.get_encoding());
}


TEST_F(FeatureSetTestFixture, FeatureSetOStream)
{
   std::cout << "\n--- FeatureSet ostream Test\n" << std::flush;

   flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>,flexnnet::RawFeature<1>>> fs;
   fs.decode({{3.14159,2.17,9.5},{666}});

   std::cout << fs << "\n";
}

TEST_F(FeatureSetTestFixture, FeatureSetIStream)
{
   std::cout << "\n--- FeatureSet istream Test\n" << std::flush;

   flexnnet::FeatureSet<std::tuple<flexnnet::RawFeature<3>,flexnnet::RawFeature<1>>> fs;

   std::string objjson = "{\n"
                         "  \"F0\":[3.14159,2.17,9.5],\n"
                         "  \"F1\":[666]\n"
                         "}";

   std::stringstream ss;

   ss.str(objjson);
   ss >> fs;

   std::cout << prettyPrintVector("raw feature 1", std::get<0>(fs.get_features()).get_encoding());
   std::cout << prettyPrintVector("raw feature 2", std::get<1>(fs.get_features()).get_encoding());
}

TEST_F(FeatureSetTestFixture, RawFeatureSet1TestConstructor)
{
   flexnnet::RawFeatureSet<1> rfs;
   EXPECT_EQ(rfs.size(), 1) << "expected 1 feature";
}

TEST_F(FeatureSetTestFixture, RawFeatureSet3TestConstructor)
{
   flexnnet::RawFeatureSet<5,3,1> rfs;
   EXPECT_EQ(rfs.size(), 3) << "expected 3 feature";
}

TEST_F(FeatureSetTestFixture, RawFeatureSet3TestSet)
{
   flexnnet::RawFeatureSet<5,3,1> rfs;
   EXPECT_EQ(rfs.size(), 3) << "expected 3 feature";

   std::get<0>(rfs.get_features()).decode({1,2,3,4,5});
   std::get<1>(rfs.get_features()).decode({3.14159,2.17,9.5});
   std::get<2>(rfs.get_features()).decode({666});

   std::cout << prettyPrintVector("raw feature 0", std::get<0>(rfs.get_features()).get_encoding());
   std::cout << prettyPrintVector("raw feature 1", std::get<1>(rfs.get_features()).get_encoding());
   std::cout << prettyPrintVector("raw feature 2", std::get<2>(rfs.get_features()).get_encoding());
}

TEST_F(FeatureSetTestFixture, ReinforcementConstructor)
{
   flexnnet::Reinforcement<1> rfs;
   EXPECT_EQ(rfs.size(), 1) << "expected 1 feature";
}

TEST_F(FeatureSetTestFixture, ReinforcementSet)
{
   flexnnet::Reinforcement<3> rfs;
   EXPECT_EQ(rfs.size(), 1) << "expected 3 feature";

   rfs.set(0, 9.5);

   std::cout << prettyPrintVector("raw feature 0", std::get<0>(rfs.get_features()).get_encoding());
}

#endif // FLEX_NEURALNET_FEATURESETTESTS_H_
