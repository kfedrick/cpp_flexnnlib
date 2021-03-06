//
// Created by kfedrick on 3/3/21.
//

#include <gtest/gtest.h>

#include "flexnnet.h"
#include "Evaluator.h"
#include "NeuralNetTopology.h"
#include "NeuralNet.h"
#include "DataSet.h"
#include "CartesianCoord.h"
#include "RMSEFitnessFunc.h"
#include "Evaluator.h"
#include "DataSetStream.h"
#include <fstream>
#include <CommonTestFixtureFunctions.h>
#include "MockNN.h"
#include <RawFeatureSet.h>

using flexnnet::DataSet;
using flexnnet::RawFeatureSet;
using flexnnet::NeuralNetTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::NeuralNet;
using flexnnet::Evaluator;
using flexnnet::RMSEFitnessFunc;
using flexnnet::CartesianCoord;
using flexnnet::Exemplar;

TEST(TestEvaluator, Constructor)
{
   std::cout << "***** Test Evaluator Constructor\n" << std::flush;

   RawFeatureSet<3> a({"a"});
   a.decode({{-1, 0, 0.5}});

   RawFeatureSet<3> b({"b"});
   b.decode({{-1, 0, 0.5}});

   RawFeatureSet<3> c({"c"});
   c.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"output"});
   c.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));

   BaseNeuralNet basenet;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(basenet);
   Evaluator<RawFeatureSet<3>, RawFeatureSet<1>, NeuralNet, DataSet, RMSEFitnessFunc> eval;
}

TEST(TestEvaluator, OrderedSingleSampling)
{
   std::cout << "***** Test Evaluator Normalized Ordering\n" << std::flush;

   RawFeatureSet<3> a({"a"});
   a.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> b({"a"});
   b.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> c({"a"});
   c.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> d({"a"});
   d.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> e({"a"});
   e.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"output"});
   c.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(d, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(e, o));

   BaseNeuralNet basenet;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(basenet);
   Evaluator<RawFeatureSet<3>, RawFeatureSet<1>, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.set_sampling_count(1);
   eval.set_subsample_fraction(0.25);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, RandomizedSingleSampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering\n" << std::flush;

   RawFeatureSet<3> a({"a"});
   a.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> b({"a"});
   b.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> c({"a"});
   c.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> d({"a"});
   d.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> e({"a"});
   e.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"output"});
   c.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(d, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(e, o));

   BaseNeuralNet basenet;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(basenet);
   Evaluator<RawFeatureSet<3>, RawFeatureSet<1>, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.randomize_order(true);
   eval.set_sampling_count(1);
   eval.set_subsample_fraction(0.33);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, Randomized2Sampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering 2 Samplings\n" << std::flush;

   RawFeatureSet<3> a({"a"});
   a.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> b({"a"});
   b.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> c({"a"});
   c.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> d({"a"});
   d.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> e({"a"});
   e.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"output"});
   c.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(d, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(e, o));

   BaseNeuralNet basenet;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(basenet);
   Evaluator<RawFeatureSet<3>, RawFeatureSet<1>, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.randomize_order(true);
   eval.set_sampling_count(2);
   eval.set_subsample_fraction(0.5);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, Randomized3SubSampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering 3 Sub-Samplings\n" << std::flush;

   RawFeatureSet<3> a({"a"});
   a.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> b({"a"});
   b.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> c({"a"});
   c.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> d({"a"});
   d.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> e({"a"});
   e.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"output"});
   c.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(d, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(e, o));

   BaseNeuralNet basenet;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(basenet);
   Evaluator<RawFeatureSet<3>, RawFeatureSet<1>, NeuralNet, DataSet, RMSEFitnessFunc> eval;

   eval.randomize_order(true);
   eval.set_sampling_count(3);
   eval.set_subsample_fraction(0.9);
   eval.evaluate(nnet, dataset);
}

TEST(TestEvaluator, BasicRMSFitNo0Egradient)
{
   std::cout << "***** Test RMSE Fitness function with no gradient\n" << std::flush;

   RMSEFitnessFunc<RawFeatureSet<3>> rmse_fit;

   RawFeatureSet<3> tst({"output"});
   RawFeatureSet<3> tgt({"output"});
   flexnnet::ValarrMap egradient;
   flexnnet::ValarrMap tgt_egradient({{"output",{0, 0, 0}}});

   tst.decode({{-1, 0, 0.5}});
   tgt.decode({{-1, 0, 0.5}});

   rmse_fit.clear();
   egradient = rmse_fit.calc_error_gradient(tgt, tst);

   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient.at("output")).c_str()
      << "\n";

   // Check egradient
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output"],
                egradient["output"], 0.000000001) << "ruh roh";

   double fitval = rmse_fit.calc_fitness();
   EXPECT_EQ(fitval, 0) << "Fitness = " << fitval << ": expected 0";
}

TEST(TestEvaluator, BasicRMSFitNoSmallEgradient)
{
   std::cout << "***** Test RMSE Fitness function with small error gradient\n" << std::flush;

   RMSEFitnessFunc<RawFeatureSet<3>> rmse_fit;

   RawFeatureSet<3> tst({"output"});
   RawFeatureSet<3> tgt({"output"});
   flexnnet::ValarrMap egradient;
   flexnnet::ValarrMap tgt_egradient({{"output",{0.03, -0.1, -0.09}}});

   tst.decode({{-1, 0, 0.5}});
   tgt.decode({{-1.03, 0.1, 0.59}});


   rmse_fit.clear();
   egradient = rmse_fit.calc_error_gradient(tgt, tst);

   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient.at("output"), 9)
         .c_str() << "\n";
   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("tgt egradient", tgt_egradient.at("output"),
                                                       9).c_str() << "\n";

   // Check basic_layer output
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output"],
                egradient["output"], 0.000000001) << "ruh roh";

   double fitval = rmse_fit.calc_fitness();
   std::cout << "RMSE = " << std::setprecision(9) << fitval << "\n" << std::flush;
   EXPECT_NEAR(fitval, 0.097467943448, 0.000000001);
}

TEST(TestEvaluator, BasicRMSFitMultiField)
{
   std::cout << "***** Test RMSE Fitness function with multiple outputs fields\n" << std::flush;

   RMSEFitnessFunc<RawFeatureSet<1, 1, 1>> rmse_fit;

   RawFeatureSet<1, 1, 1> tst({"output1", "output2", "output3"});
   RawFeatureSet<1, 1, 1> tgt({"output1", "output2", "output3"});
   flexnnet::ValarrMap egradient;
   flexnnet::ValarrMap tgt_egradient({{"output1",{0.03}}, {"output2",{-0.1}}, {"output3",{-0.09}}});

   tst.decode({{-1}, {0}, {0.5}});
   tgt.decode({{-1.03}, {0.1}, {0.59}});

   rmse_fit.clear();
   egradient = rmse_fit.calc_error_gradient(tgt, tst);

   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient["output1"], 9).c_str()
      << "\n";
   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient["output2"], 9).c_str()
      << "\n";
   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient["output3"], 9).c_str()
      << "\n";

   // Check basic_layer output
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output1"],
                egradient["output1"], 0.000000001) << "ruh roh";
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output2"],
                egradient["output2"], 0.000000001) << "ruh roh";
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output3"],
                egradient["output3"], 0.000000001) << "ruh roh";

   double fitval = rmse_fit.calc_fitness();
   EXPECT_NEAR(fitval, 0.097467943448, 0.000000001);
}

TEST(TestEvaluator, BasicRMSFitMultiSample)
{
   std::cout << "***** Test RMSE Fitness function with multiple samples\n" << std::flush;

   RMSEFitnessFunc<RawFeatureSet<3>> rmse_fit;

   RawFeatureSet<3> tst1({"output"});
   RawFeatureSet<3> tgt1({"output"});
   flexnnet::ValarrMap egradient1;
   flexnnet::ValarrMap tgt_egradient1({{"output",{0.03, -0.1, -0.09}}});
   RawFeatureSet<3> tst2({"output"});
   RawFeatureSet<3> tgt2({"output"});
   flexnnet::ValarrMap egradient2;
   flexnnet::ValarrMap tgt_egradient2({{"output",{0.33, -1.2, 0.285}}});

   tst1.decode({{-1, 0, 0.5}});
   tgt1.decode({{-1.03, 0.1, 0.59}});
   tst2.decode({{-0.3, -1.3, 0.875}});
   tgt2.decode({{-0.63, -0.1, 0.59}});


   rmse_fit.clear();
   egradient1 = rmse_fit.calc_error_gradient(tgt1, tst1);
   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient1", egradient1.at("output"), 9)
         .c_str() << "\n";

   // Check basic_layer output
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient1["output"],
                egradient1["output"], 0.000000001) << "ruh roh";

   double fitval1 = rmse_fit.calc_fitness();
   std::cout << "RMSE1 = " << std::setprecision(9) << fitval1 << "\n" << std::flush;
   EXPECT_NEAR(fitval1, 0.097467943448, 0.000000001);

   egradient2 = rmse_fit.calc_error_gradient(tgt2, tst2);
   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient2", egradient2.at("output"), 9)
         .c_str() << "\n";

   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient2["output"],
                egradient2["output"], 0.000000001) << "ruh roh";

   double fitval2 = rmse_fit.calc_fitness();
   std::cout << "RMSE2 = " << std::setprecision(9) << fitval2 << "\n" << std::flush;
   EXPECT_NEAR(fitval2, 0.6420913097, 0.000000001);
}

TEST(TestEvaluator, BasicRMSEEvaluatorTest)
{
   std::cout << "***** Test Basic RMSFitness Evaluator\n" << std::flush;

   RawFeatureSet<3> tst1({"output"});
   RawFeatureSet<3> tgt1({"output"});
   flexnnet::ValarrMap egradient1;
   flexnnet::ValarrMap tgt_egradient1({{"output",{0.0003, 0.00333333333, 0.0027}}});
   RawFeatureSet<3> tst2({"output"});
   RawFeatureSet<3> tgt2({"output"});
   flexnnet::ValarrMap egradient2;
   flexnnet::ValarrMap tgt_egradient2({{"output", {0.0363, 0.48, 0.027075}}});

   tst1.decode({{-1, 0, 0.5}});
   tgt1.decode({{-1.03, 0.1, 0.59}});
   tst2.decode({{-0.3, -1.3, 0.875}});
   tgt2.decode({{-0.63, -0.1, 0.59}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<3>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<3>>(tst1, tgt1));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<3>>(tst2, tgt2));

   RMSEFitnessFunc<RawFeatureSet<3>> rmse_fit;
   BaseNeuralNet basenet;
   MockNN<RawFeatureSet<3>, RawFeatureSet<3>> nnet(basenet);
   Evaluator<RawFeatureSet<3>, RawFeatureSet<3>, MockNN, DataSet, RMSEFitnessFunc> eval;

   eval.set_sampling_count(1);
   eval.set_subsample_fraction(1.0);

   double rmse, errstd;
   std::tie(rmse, errstd) =
      eval.evaluate(reinterpret_cast<MockNN<RawFeatureSet<3>, RawFeatureSet<3>>&>(nnet), dataset);

   std::cout << rmse << ", " << errstd << "\n" << std::flush;
   EXPECT_NEAR(rmse, 0.64209131, 0.000000001) << "Bad mean fitness score.\n";
   EXPECT_NEAR(errstd, 0, 0.000000001) << "Bad fitness score standard dev.\n";
}

TEST(TestEvaluator, SubsampledRMSEEvaluatorTest)
{
   std::cout << "***** Test RMSFitness Evaluator with subsampling\n" << std::flush;

   RawFeatureSet<3> tst1({"output"});
   RawFeatureSet<3> tgt1({"output"});
   flexnnet::ValarrMap egradient1;
   flexnnet::ValarrMap tgt_egradient1({{"output",{0.0003, 0.00333333333, 0.0027}}});
   RawFeatureSet<3> tst2({"output"});
   RawFeatureSet<3> tgt2({"output"});
   flexnnet::ValarrMap egradient2;
   flexnnet::ValarrMap tgt_egradient2({{"output",{0.0363, 0.48, 0.027075}}});
   RawFeatureSet<3> tst3({"output"});
   RawFeatureSet<3> tgt3({"output"});
   flexnnet::ValarrMap egradient3;
   flexnnet::ValarrMap tgt_egradient3({{"output",{0.0363, 0.48, 0.027075}}});

   tst1.decode({{-1, 0, 0.5}});
   tgt1.decode({{-1.03, 0.1, 0.59}});

   tst2.decode({{-0.3, -1.3, 0.875}});
   tgt2.decode({{-0.63, -0.1, 0.59}});

   tst3.decode({{0.87, -0.326, 0.001}});
   tgt3.decode({{0.59, -0.1, 0.59}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<3>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<3>>(tst1, tgt1));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<3>>(tst2, tgt2));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<3>>(tst3, tgt3));

   RMSEFitnessFunc<RawFeatureSet<3>> rmse_fit;
   BaseNeuralNet basenet;
   MockNN<RawFeatureSet<3>, RawFeatureSet<3>> nnet(basenet);
   Evaluator<RawFeatureSet<3>, RawFeatureSet<3>, MockNN, DataSet, RMSEFitnessFunc> eval;

   eval.set_sampling_count(1000);
   eval.randomize_order(true);
   eval.set_subsample_fraction(0.67);

   double rmse, errstd;
   std::tie(rmse, errstd) =
      eval.evaluate(reinterpret_cast<MockNN<RawFeatureSet<3>, RawFeatureSet<3>>&>(nnet), dataset);

   std::cout << rmse << ", " << errstd << "\n" << std::flush;
   EXPECT_NEAR(rmse, 0.573, 0.01) << "Bad mean fitness score.\n";
   EXPECT_NEAR(errstd, 0.16, 0.01) << "Bad fitness score standard dev.\n";
}

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   srand(time(NULL));

   return RUN_ALL_TESTS();
}