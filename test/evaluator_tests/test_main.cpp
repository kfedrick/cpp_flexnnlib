//
// Created by kfedrick on 3/3/21.
//

#include <gtest/gtest.h>

#include "flexnnet.h"
#include "Evaluator.h"
#include "NeuralNet.h"
#include "DataSet.h"
#include "CartesianCoord.h"
#include "RMSEFitnessFunc.h"
#include "Evaluator.h"
#include "DataSetStream.h"
#include <ValarrayMap.h>
#include <fstream>
#include <CommonTestFixtureFunctions.h>
#include "MockNN.h"

using flexnnet::DataSet;
using flexnnet::NetworkTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::NeuralNet;
using flexnnet::Evaluator;
using flexnnet::RMSEFitnessFunc;
using flexnnet::CartesianCoord;
using flexnnet::ValarrayMap;

TEST(TestEvaluator, Constructor)
{
   std::cout << "***** Test Evaluator Constructor\n" << std::flush;

   ValarrayMap a({{"a",{-1, 0, 0.5}}});
   ValarrayMap b({{"b",{-1, 0, 0.5}}});
   ValarrayMap c({{"c",{-1, 0, 0.5}}});

   DataSet<ValarrayMap, ValarrayMap> dataset;
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(a, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(b, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(c, {}));

   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<ValarrayMap, ValarrayMap> nnet(basenet);
   Evaluator<ValarrayMap, ValarrayMap, NeuralNet, DataSet, RMSEFitnessFunc> eval;
}

TEST(TestEvaluator, OrderedSingleSampling)
{
   std::cout << "***** Test Evaluator Normalized Ordering\n" << std::flush;

   ValarrayMap a({{"a",{-1, 0, 0.5}}});
   ValarrayMap b({{"b",{-1, 0, 0.5}}});
   ValarrayMap c({{"c",{-1, 0, 0.5}}});
   ValarrayMap d({{"d",{-1, 0, 0.5}}});
   ValarrayMap e({{"e",{-1, 0, 0.5}}});

   DataSet<ValarrayMap, ValarrayMap> dataset;
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(a, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(b, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(c, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(d, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(e, {}));

   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<ValarrayMap, ValarrayMap> nnet(basenet);
   Evaluator<ValarrayMap, ValarrayMap, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.set_sampling_count(1);
   eval.set_subsample_fraction(0.25);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, RandomizedSingleSampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering\n" << std::flush;

   ValarrayMap a({{"a",{-1, 0, 0.5}}});
   ValarrayMap b({{"b",{-1, 0, 0.5}}});
   ValarrayMap c({{"c",{-1, 0, 0.5}}});
   ValarrayMap d({{"d",{-1, 0, 0.5}}});
   ValarrayMap e({{"e",{-1, 0, 0.5}}});

   DataSet<ValarrayMap, ValarrayMap> dataset;
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(a, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(b, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(c, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(d, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(e, {}));

   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<ValarrayMap, ValarrayMap> nnet(basenet);
   Evaluator<ValarrayMap, ValarrayMap, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.randomize_order(true);
   eval.set_sampling_count(1);
   eval.set_subsample_fraction(0.33);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, Randomized2Sampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering 2 Samplings\n" << std::flush;

   ValarrayMap a({{"a",{-1, 0, 0.5}}});
   ValarrayMap b({{"b",{-1, 0, 0.5}}});
   ValarrayMap c({{"c",{-1, 0, 0.5}}});
   ValarrayMap d({{"d",{-1, 0, 0.5}}});
   ValarrayMap e({{"e",{-1, 0, 0.5}}});

   DataSet<ValarrayMap, ValarrayMap> dataset;
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(a, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(b, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(c, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(d, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(e, {}));

   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<ValarrayMap, ValarrayMap> nnet(basenet);
   Evaluator<ValarrayMap, ValarrayMap, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.randomize_order(true);
   eval.set_sampling_count(2);
   eval.set_subsample_fraction(0.5);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, Randomized3SubSampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering 3 Sub-Samplings\n" << std::flush;

   ValarrayMap a({{"a",{-1, 0, 0.5}}});
   ValarrayMap b({{"b",{-1, 0, 0.5}}});
   ValarrayMap c({{"c",{-1, 0, 0.5}}});
   ValarrayMap d({{"d",{-1, 0, 0.5}}});
   ValarrayMap e({{"e",{-1, 0, 0.5}}});

   DataSet<ValarrayMap, ValarrayMap> dataset;
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(a, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(b, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(c, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(d, {}));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(e, {}));

   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<ValarrayMap, ValarrayMap> nnet(basenet);
   Evaluator<ValarrayMap, ValarrayMap, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.randomize_order(true);
   eval.set_sampling_count(3);
   eval.set_subsample_fraction(0.9);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, CartesianCoord)
{
   std::cout << "***** Test Derived Evaluator CartesianCoord\n" << std::flush;

   DataSet<CartesianCoord, ValarrayMap> dataset;
   BaseNeuralNet basenet(NetworkTopology({}));
   NeuralNet<CartesianCoord, ValarrayMap> nnet(basenet);
   Evaluator<CartesianCoord, ValarrayMap, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.set_sampling_count(1);
   eval.set_subsample_fraction(0.5);
   eval.evaluate(nnet, dataset);
}


TEST(TestEvaluator, BasicRMSFitNo0Egradient)
{
   std::cout << "***** Test RMSE Fitness function with no gradient\n" << std::flush;

   RMSEFitnessFunc<ValarrayMap> rmse_fit;

   ValarrayMap tst;
   ValarrayMap tgt;
   ValarrayMap egradient;
   ValarrayMap tgt_egradient;

   tst = {{"output",{-1, 0, 0.5}}};
   tgt = {{"output",{-1, 0, 0.5}}};
   tgt_egradient = {{"output",{0, 0, 0}}};

   rmse_fit.clear();
   egradient = rmse_fit.calc_error_gradient(tgt,tst);

   std::cout << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient.at("output")).c_str() << "\n";

   // Check egradient
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output"], egradient["output"], 0.000000001) << "ruh roh";

   double fitval = rmse_fit.calc_fitness();
   EXPECT_EQ(fitval, 0) << "Fitness = " << fitval << ": expected 0";
}

TEST(TestEvaluator, BasicRMSFitNoSmallEgradient)
{
   std::cout << "***** Test RMSE Fitness function with small error gradient\n" << std::flush;

   RMSEFitnessFunc<ValarrayMap> rmse_fit;

   ValarrayMap tst;
   ValarrayMap tgt;
   ValarrayMap egradient;
   ValarrayMap tgt_egradient;

   tst = {{"output",{-1, 0, 0.5}}};
   tgt = {{"output",{-1.03, 0.1, 0.59}}};
   tgt_egradient = {{"output",{0.0003, 0.00333333333, 0.0027}}};

   rmse_fit.clear();
   egradient = rmse_fit.calc_error_gradient(tgt,tst);

   std::cout << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient.at("output"), 9).c_str() << "\n";

   // Check basic_layer output
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output"], egradient["output"], 0.000000001) << "ruh roh";

   double fitval = rmse_fit.calc_fitness();
   std::cout << "RMSE = " << std::setprecision(9) << fitval << "\n" << std::flush;
   EXPECT_NEAR(fitval, 0.0795822426, 0.000000001);
}

TEST(TestEvaluator, BasicRMSFitMultiField)
{
   std::cout << "***** Test RMSE Fitness function with multiple outputs fields\n" << std::flush;

   RMSEFitnessFunc<ValarrayMap> rmse_fit;

   ValarrayMap tst;
   ValarrayMap tgt;
   ValarrayMap egradient;
   ValarrayMap tgt_egradient;

   tst = {{"output1",{-1}}, {"output2",{0}}, {"output3",{0.5}}};
   tgt = {{"output1",{-1.03}}, {"output2",{0.1}}, {"output3",{0.59}}};
   tgt_egradient = {{"output1",{0.0003}}, {"output2",{0.00333333333}}, {"output3",{0.0027}}};

   rmse_fit.clear();
   egradient = rmse_fit.calc_error_gradient(tgt,tst);

   std::cout << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient["output1"], 9).c_str() << "\n";
   std::cout << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient["output2"], 9).c_str() << "\n";
   std::cout << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient["output3"], 9).c_str() << "\n";

   // Check basic_layer output
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output1"], egradient["output1"], 0.000000001) << "ruh roh";
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output2"], egradient["output2"], 0.000000001) << "ruh roh";
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output3"], egradient["output3"], 0.000000001) << "ruh roh";

   double fitval = rmse_fit.calc_fitness();
   EXPECT_NEAR(fitval, 0.0795822426, 0.000000001);
}

TEST(TestEvaluator, BasicRMSFitMultiSample)
{
   std::cout << "***** Test RMSE Fitness function with multiple samples\n" << std::flush;

   RMSEFitnessFunc<ValarrayMap> rmse_fit;

   ValarrayMap tst1, tst2;
   ValarrayMap tgt1, tgt2;
   ValarrayMap egradient1, egradient2;
   ValarrayMap tgt_egradient1, tgt_egradient2;

   tst1 = {{"output",{-1, 0, 0.5}}};
   tgt1 = {{"output",{-1.03, 0.1, 0.59}}};
   tgt_egradient1 = {{"output",{0.0003, 0.00333333333, 0.0027}}};
   tst2 = {{"output",{-0.3, -1.3, 0.875}}};
   tgt2 = {{"output",{-0.63, -0.1, 0.59}}};
   tgt_egradient2 = {{"output",{0.0363, 0.48, 0.027075}}};

   rmse_fit.clear();
   egradient1 = rmse_fit.calc_error_gradient(tgt1,tst1);
   std::cout << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient1.at("output"), 9).c_str() << "\n";

   // Check basic_layer output
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient1["output"], egradient1["output"], 0.000000001) << "ruh roh";

   double fitval1 = rmse_fit.calc_fitness();
   std::cout << "RMSE1 = " << std::setprecision(9) << fitval1 << "\n" << std::flush;
   EXPECT_NEAR(fitval1, 0.0795822426, 0.000000001);

   egradient2 = rmse_fit.calc_error_gradient(tgt2,tst2);
   std::cout << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient2.at("output"), 9).c_str() << "\n";

   double fitval2 = rmse_fit.calc_fitness();
   std::cout << "RMSE2 = " << std::setprecision(9) << fitval2 << "\n" << std::flush;
   EXPECT_NEAR(fitval2, 0.524265359, 0.000000001);

   rmse_fit.clear();
   egradient2 = rmse_fit.calc_error_gradient(tgt2,tst2);
   std::cout << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient2.at("output"), 9).c_str() << "\n";

   double fitval3 = rmse_fit.calc_fitness();
   std::cout << "RMSE3 = " << std::setprecision(10) << fitval3 << "\n" << std::flush;
   EXPECT_NEAR(fitval3, 0.7371397425, 0.000000001);
}

TEST(TestEvaluator, BasicRMSEEvaluatorTest)
{
   std::cout << "***** Test Basic RMSFitness Evaluator\n" << std::flush;

   ValarrayMap tst1({{"output",{-1, 0, 0.5}}});
   ValarrayMap tgt1({{"output",{-1.03, 0.1, 0.59}}});
   ValarrayMap tgt_egradient1({{"output",{0.0003, 0.00333333333, 0.0027}}});
   ValarrayMap tst2({{"output",{-0.3, -1.3, 0.875}}});
   ValarrayMap tgt2({{"output",{-0.63, -0.1, 0.59}}});
   ValarrayMap tgt_egradient2({{"output",{0.0363, 0.48, 0.027075}}});

   DataSet<ValarrayMap, ValarrayMap> dataset;
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(tst1, tgt1));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(tst2, tgt2));

   RMSEFitnessFunc<ValarrayMap> rmse_fit;
   BaseNeuralNet basenet(NetworkTopology({}));
   MockNN<ValarrayMap,ValarrayMap> nnet(basenet);
   Evaluator<ValarrayMap, ValarrayMap, MockNN, DataSet, RMSEFitnessFunc> eval;

   eval.set_sampling_count(1);
   eval.set_subsample_fraction(0.5);

   double rmse, errstd;
   std::tie(rmse, errstd) = eval.evaluate(reinterpret_cast<MockNN<ValarrayMap,ValarrayMap>&>(nnet), dataset);

   std::cout << rmse << ", " << errstd << "\n" << std::flush;
   EXPECT_NEAR(rmse, 0.07958224258, 0.000000001) << "Bad mean fitness score.\n";
   EXPECT_NEAR(errstd, 0, 0.000000001) << "Bad fitness score standard dev.\n";
}

TEST(TestEvaluator, SubsampledRMSEEvaluatorTest)
{
   std::cout << "***** Test RMSFitness Evaluator with subsampling\n" << std::flush;

   ValarrayMap tst1({{"output",{-1, 0, 0.5}}});
   ValarrayMap tgt1({{"output",{-1.03, 0.1, 0.59}}});
   ValarrayMap tgt_egradient1({{"output",{0.0003, 0.00333333333, 0.0027}}});

   ValarrayMap tst2({{"output",{-0.3, -1.3, 0.875}}});
   ValarrayMap tgt2({{"output",{-0.63, -0.1, 0.59}}});
   ValarrayMap tgt_egradient2({{"output",{0.0363, 0.48, 0.027075}}});

   ValarrayMap tst3({{"output",{0.87, -0.326, 0.001}}});
   ValarrayMap tgt3({{"output",{0.59, -0.1, 0.59}}});
   ValarrayMap tgt_egradient3({{"output",{0.0363, 0.48, 0.027075}}});

   DataSet<ValarrayMap, ValarrayMap> dataset;
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(tst1, tgt1));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(tst2, tgt2));
   dataset.push_back(std::pair<ValarrayMap, ValarrayMap>(tst3, tgt3));

   RMSEFitnessFunc<ValarrayMap> rmse_fit;
   BaseNeuralNet basenet(NetworkTopology({}));
   MockNN<ValarrayMap,ValarrayMap> nnet(basenet);
   Evaluator<ValarrayMap, ValarrayMap, MockNN, DataSet, RMSEFitnessFunc> eval;

   eval.set_sampling_count(17);
   eval.randomize_order(true);
   eval.set_subsample_fraction(0.67);

   double rmse, errstd;
   std::tie(rmse, errstd) = eval.evaluate(reinterpret_cast<MockNN<ValarrayMap,ValarrayMap>&>(nnet), dataset);

   std::cout << rmse << ", " << errstd << "\n" << std::flush;
}

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   srand (time(NULL));

   return RUN_ALL_TESTS();
}