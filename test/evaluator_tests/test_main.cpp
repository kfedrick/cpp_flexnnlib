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
#include "DataSetStream.h"
#include <fstream>
#include <CommonTestFixtureFunctions.h>
#include "MockNN.h"
#include <RawFeatureSet.h>
#include <NetworkLayerImpl.h>
#include <TanSig.h>

using flexnnet::DataSet;
using flexnnet::RawFeatureSet;
using flexnnet::NeuralNetTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::NeuralNet;
using flexnnet::CartesianCoord;
using flexnnet::Exemplar;
using flexnnet::RMSEFitnessFunc;
using flexnnet::NetworkLayerImpl;
using flexnnet::TanSig;

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

   NeuralNetTopology topo;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(topo);
   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> lossfunc;
}

TEST(TestEvaluator, OrderedSingleSampling)
{
   std::cout << "***** Test Evaluator Normalized Ordering\n" << std::flush;

   RawFeatureSet<3> a({"F0"});
   a.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> b({"F0"});
   b.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> c({"F0"});
   c.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> d({"F0"});
   d.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> e({"F0"});
   e.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"F0"});
   o.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(d, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(e, o));

   NeuralNetTopology topo;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(topo);
   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> lossfunc;

   lossfunc.calc_fitness(nnet, dataset, 0.25);
}

TEST(TestEvaluator, RandomizedSingleSampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering\n" << std::flush;

   RawFeatureSet<3> a({"F0"});
   a.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> b({"F0"});
   b.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> c({"F0"});
   c.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> d({"F0"});
   d.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> e({"F0"});
   e.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"F0"});
   c.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(d, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(e, o));

   NeuralNetTopology topo;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(topo);
   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> lossfunc;

   lossfunc.calc_fitness(nnet, dataset, 0.33);
}

TEST(TestEvaluator, Randomized2Sampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering 2 Samplings\n" << std::flush;

   RawFeatureSet<3> a({"F0"});
   a.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> b({"F0"});
   b.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> c({"F0"});
   c.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> d({"F0"});
   d.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> e({"F0"});
   e.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"F0"});
   c.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(d, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(e, o));

   NeuralNetTopology topo;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(topo);
   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> lossfunc;

   //eval.set_sampling_count(2);
   lossfunc.calc_fitness(nnet, dataset, 0.5);
}

TEST(TestEvaluator, Randomized3SubSampling)
{
   std::cout << "***** Test Evaluator Randomized Ordering 3 Sub-Samplings\n" << std::flush;

   RawFeatureSet<3> a({"F0"});
   a.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> b({"F0"});
   b.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> c({"F0"});
   c.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> d({"F0"});
   d.decode({{-1, 0, 0.5}});
   RawFeatureSet<3> e({"F0"});
   e.decode({{-1, 0, 0.5}});

   RawFeatureSet<1> o({"F0"});
   c.decode({{0.5}});

   DataSet<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> dataset;
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(a, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(b, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(c, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(d, o));
   dataset.push_back(Exemplar<RawFeatureSet<3>, RawFeatureSet<1>>(e, o));

   NeuralNetTopology topo;
   NeuralNet<RawFeatureSet<3>, RawFeatureSet<1>> nnet(topo);
   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<1>, Exemplar> lossfunc;

   //eval.set_sampling_count(3);
   lossfunc.calc_fitness(nnet, dataset, 0.9);
}

TEST(TestEvaluator, BasicRMSFitNo0Egradient)
{
   std::cout << "***** Test RMSE Fitness function with no gradient\n" << std::flush;

   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<3>, Exemplar> lossfunc;

   RawFeatureSet<3> tst({"output"});
   RawFeatureSet<3> tgt({"output"});
   flexnnet::ValarrMap egradient;
   flexnnet::ValarrMap tgt_egradient({{"output",{0, 0, 0}}});

   tst.decode({{-1, 0, 0.5}});
   tgt.decode({{-1, 0, 0.5}});

   double err;
   err = lossfunc.calc_dEde(tgt, tst, egradient);

   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient.at("output")).c_str()
      << "\n";

   // Check egradient
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output"],
                egradient["output"], 0.000000001) << "ruh roh";

   //double fitval = lossfunc.calc_fitness();
   //EXPECT_EQ(fitval, 0) << "Fitness = " << fitval << ": expected 0";
}

TEST(TestEvaluator, BasicRMSFitNoSmallEgradient)
{
   std::cout << "***** Test RMSE Fitness function with small error gradient\n" << std::flush;

   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<3>, Exemplar> lossfunc;

   RawFeatureSet<3> tst({"output"});
   RawFeatureSet<3> tgt({"output"});
   flexnnet::ValarrMap egradient;
   flexnnet::ValarrMap tgt_egradient({{"output",{0.03, -0.1, -0.09}}});

   tst.decode({{-1, 0, 0.5}});
   tgt.decode({{-1.03, 0.1, 0.59}});

   double err;
   err = lossfunc.calc_dEde(tgt, tst, egradient);

   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient", egradient.at("output"), 9)
         .c_str() << "\n";
   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("tgt egradient", tgt_egradient.at("output"),
                                                       9).c_str() << "\n";

   // Check basic_layer output
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient["output"],
                egradient["output"], 0.000000001) << "ruh roh";

   //double fitval = lossfunc.calc_fitness();
   //std::cout << "RMSE = " << std::setprecision(9) << fitval << "\n" << std::flush;
   //EXPECT_NEAR(fitval, 0.097467943448, 0.000000001);
}

TEST(TestEvaluator, BasicRMSFitMultiField)
{
   std::cout << "***** Test RMSE Fitness function with multiple outputs fields\n" << std::flush;

   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<1,1,1>, Exemplar> lossfunc;

   RawFeatureSet<1, 1, 1> tst({"output1", "output2", "output3"});
   RawFeatureSet<1, 1, 1> tgt({"output1", "output2", "output3"});
   flexnnet::ValarrMap egradient;
   flexnnet::ValarrMap tgt_egradient({{"output1",{0.03}}, {"output2",{-0.1}}, {"output3",{-0.09}}});

   tst.decode({{-1}, {0}, {0.5}});
   tgt.decode({{-1.03}, {0.1}, {0.59}});

   double err;
   err = lossfunc.calc_dEde(tgt, tst, egradient);

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

   //double fitval = lossfunc.calc_fitness();
   //EXPECT_NEAR(fitval, 0.097467943448, 0.000000001);
}

TEST(TestEvaluator, BasicRMSFitMultiSample)
{
   std::cout << "***** Test RMSE Fitness function with multiple samples\n" << std::flush;

   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<3>, Exemplar> lossfunc;

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

   double err1, err2;
   err1 = lossfunc.calc_dEde(tgt1, tst1, egradient1);
   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient1", egradient1.at("output"), 9)
         .c_str() << "\n";

   // Check basic_layer output
   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient1["output"],
                egradient1["output"], 0.000000001) << "ruh roh";

   //double fitval1 = lossfunc.calc_fitness();
   //std::cout << "RMSE1 = " << std::setprecision(9) << fitval1 << "\n" << std::flush;
   //EXPECT_NEAR(fitval1, 0.097467943448, 0.000000001);

   err2 = lossfunc.calc_dEde(tgt2, tst2, egradient2);
   std::cout
      << CommonTestFixtureFunctions::prettyPrintVector("egradient2", egradient2.at("output"), 9)
         .c_str() << "\n";

   EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, tgt_egradient2["output"],
                egradient2["output"], 0.000000001) << "ruh roh";

   //double fitval2 = lossfunc.calc_fitness();
   //std::cout << "RMSE2 = " << std::setprecision(9) << fitval2 << "\n" << std::flush;
   //EXPECT_NEAR(fitval2, 0.6420913097, 0.000000001);
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


   NeuralNetTopology topo;
   MockNN<RawFeatureSet<3>, RawFeatureSet<3>> nnet(topo);
   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<3>, Exemplar> lossfunc;

   lossfunc.set_subsample_count(3);

   double rmse, errstd;
   //std::tie(rmse, errstd) =
   //   eval.evaluate(reinterpret_cast<MockNN<RawFeatureSet<3>, RawFeatureSet<3>>&>(nnet), dataset);
   std::tie(rmse, errstd) = lossfunc.calc_fitness_standard_error(nnet, dataset, 1.0);
   //rmse = lossfunc.calc_fitness(nnet, dataset);

   std::cout << rmse << ", " << errstd << "\n" << std::flush;

   //double rmse_loss = rmse_loss_func.calc_fitness(nnet, dataset, 1.0);
   //std::cout << "rmse loss function value = " << rmse_loss << "\n" << std::flush;


   EXPECT_NEAR(rmse, 0.64209131, 0.000000001) << "Bad mean fitness score.\n";
   //EXPECT_NEAR(errstd, 0, 0.000000001) << "Bad fitness score standard dev.\n";
}

TEST(TestEvaluator, SubsampledRMSEEvaluatorTest)
{
   std::cout << "***** Test RMSFitness Evaluator with subsampling\n" << std::flush;

   RawFeatureSet<3> tst1({"input"});
   RawFeatureSet<3> tgt1({"output"});
   flexnnet::ValarrMap egradient1;
   flexnnet::ValarrMap tgt_egradient1({{"output",{0.0003, 0.00333333333, 0.0027}}});
   RawFeatureSet<3> tst2({"input"});
   RawFeatureSet<3> tgt2({"output"});
   flexnnet::ValarrMap egradient2;
   flexnnet::ValarrMap tgt_egradient2({{"output",{0.0363, 0.48, 0.027075}}});
   RawFeatureSet<3> tst3({"input"});
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

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr = std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(3, "output", TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 3);

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   MockNN<RawFeatureSet<3>, RawFeatureSet<3>> nnet(topo);
   RMSEFitnessFunc<RawFeatureSet<3>, RawFeatureSet<3>, Exemplar> lossfunc;

   lossfunc.set_subsample_count(1000);

   double rmse, errstd;
   std::tie(rmse,errstd) = lossfunc.calc_fitness_standard_error(nnet, dataset, 0.67);

   std::cout << rmse << ", " << errstd << "\n" << std::flush;
   EXPECT_NEAR(rmse, 0.573, 0.01) << "Bad mean fitness score.\n";
   //EXPECT_NEAR(errstd, 0.16, 0.01) << "Bad fitness score standard dev.\n";
}


int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   srand(time(NULL));

   return RUN_ALL_TESTS();
}