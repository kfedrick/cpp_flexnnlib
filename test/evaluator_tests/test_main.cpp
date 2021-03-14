//
// Created by kfedrick on 3/3/21.
//

#include <gtest/gtest.h>

#include "flexnnet.h"
#include "Evaluator.h"
#include "DerivedEvaluator.h"
#include "NeuralNet.h"
#include "EnumeratedDataSet.h"
#include "CartesianCoord.h"
#include <ValarrayMap.h>

using flexnnet::EnumeratedDataSet;
using flexnnet::NetworkTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::NeuralNet;
using flexnnet::Evaluator;
using flexnnet::DerivedEvaluator;
using flexnnet::FitnessFunction;
using flexnnet::CartesianCoord;
using flexnnet::ValarrayMap;

TEST(TestEvaluator, Constructor)
{
   std::cout << "Test Evaluator Constructor\n" << std::flush;

   EnumeratedDataSet<ValarrayMap, ValarrayMap> dataset;
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>({}, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>({}, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>({}, {}));

   std::cout << "data set size = " << dataset.size() << "\n" << std::flush;

   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<ValarrayMap, ValarrayMap> nnet(basenet);
   Evaluator<ValarrayMap, ValarrayMap, NeuralNet, EnumeratedDataSet, FitnessFunction> eval(dataset);

   eval.set_sampling_count(10);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, DerivedConstructor)
{
   std::cout << "Test Derived Evaluator Constructor\n" << std::flush;

   EnumeratedDataSet<ValarrayMap, ValarrayMap> dataset;
   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<ValarrayMap, ValarrayMap> nnet(basenet);
   DerivedEvaluator<ValarrayMap, ValarrayMap, NeuralNet, EnumeratedDataSet> eval(dataset);

   eval.set_sampling_count(10);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, CartesianCoord)
{
   std::cout << "Test Derived Evaluator CartesianCoord\n" << std::flush;

   EnumeratedDataSet<CartesianCoord, ValarrayMap> dataset;
   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<CartesianCoord, ValarrayMap> nnet(basenet);
   DerivedEvaluator<CartesianCoord, ValarrayMap, NeuralNet, EnumeratedDataSet> eval(dataset);

   eval.set_sampling_count(10);
   eval.evaluate(nnet, dataset);
}

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);

   return RUN_ALL_TESTS();
}