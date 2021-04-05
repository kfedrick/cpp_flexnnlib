//
// Created by kfedrick on 3/29/21.
//

#ifndef _CLASSIFIERTRAININGTESTS_H_
#define _CLASSIFIERTRAININGTESTS_H_

#include "gtest/gtest.h"

#include "SupervisedTrainerTestFixture.h"

#include "flexnnet.h"
#include "Evaluator.h"
#include "NetworkTopology.h"
#include "NeuralNet.h"
#include "DataSet.h"
#include "RMSEFitnessFunc.h"
#include "Evaluator.h"
#include <ValarrayMap.h>
#include <fstream>
#include <CommonTestFixtureFunctions.h>
#include <SuperviseTrainingAlgo.h>
#include <PureLin.h>
#include <DeltaBarDeltaLearningRate.h>
#include <TanSig.h>
#include <ConstantLearningRate.h>
#include <SoftMax.h>
#include <LogSig.h>
#include <RadBas.h>
#include "SimpleBinaryClassifierDataSet.h"

TEST_F (SupervisedTrainerTestFixture, SingleLinBinClassifierTrainingTest)
{
   /*
    * Generate the training data
    */
   const double MEAN_A = 2.0;
   const double STD_A = 0.01;

   const double MEAN_B = -2.0;
   const double STD_B = 0.025;

   const double ERROR_GOAL = 0.02;

   SimpleBinaryClassifierDataSet trnset;

   trnset.generate_samples(43, 0, MEAN_A, STD_A);
   trnset.generate_samples(51, 1, MEAN_B, STD_B);
   EXPECT_NO_THROW("Unexpected exception thrown while building training data set.");


   SimpleBinaryClassifierDataSet tstset;

   trnset.generate_samples(243, 0, MEAN_A, STD_A);
   trnset.generate_samples(251, 1, MEAN_B, STD_B);
   EXPECT_NO_THROW("Unexpected exception thrown while building training data set.");

   /*
    * Set up RMSE fitness and performance evaluator.
    */
   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;

   /*
    * Configure NN topology and base neural net
    */
   flexnnet::ValarrMap nninput;
   nninput["input"] = {1};
   flexnnet::NetworkTopology nntopo(nninput);
   nntopo.add_layer<flexnnet::PureLin>("output", 1, true);
   nntopo.add_external_input_field("output", "input");
   EXPECT_NO_THROW("Unexpected exception thrown while building NetworkTopology.");

   flexnnet::BaseNeuralNet basenet(nntopo);
   EXPECT_NO_THROW("Unexpected exception thrown while defining BaseNeuralNet.");

   /*
    * Define templatized neural network
    */
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       NeuralNet,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   EXPECT_NO_THROW("Unexpected exception thrown while defining NN<>.");

   /*
    * Define and configure trainer
    */
   flexnnet::SuperviseTrainingAlgo<flexnnet::ValarrayMap,
                                   flexnnet::ValarrayMap,
                                   NeuralNet,
                                   flexnnet::DataSet,
                                   flexnnet::Evaluator,
                                   flexnnet::RMSEFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer(nnet);

   trainer.set_saved_nnet_limit(20);
   trainer.set_batch_mode(0);
   trainer.set_training_runs(3);
   trainer.set_max_epochs(100);
   trainer.set_learning_rate(0.0001);
   trainer.set_error_goal(ERROR_GOAL);
   //trainer.set_train_biases("output", false);

   EXPECT_NO_THROW("Unexpected exception thrown while configuring trainer.");

   flexnnet::NetworkWeights nw = nnet.get_weights();
   flexnnet::LayerWeights iow = nw.at("output");

   trainer.train(trnset);
   EXPECT_NO_THROW("Unexpected exception thrown while training network.");

   nw = nnet.get_weights();
   flexnnet::LayerWeights tow = nw.at("output");

/*
   std::cout << prettyPrintArray("initial output weights", iow.const_weights_ref);

   std::cout << prettyPrintArray("trained output weights", tow.const_weights_ref);

*/

   const flexnnet::TrainingReport& tr = trainer.get_training_report();

   std::cout << "\ntotal training runs = " << tr.total_training_runs() << "\n";

   double best_perf_mean, perf_stdev;
   double success_rate;

   success_rate = tr.successful_training_rate();
   std::tie<double,double>(best_perf_mean, perf_stdev) = tr.performance_statistics();
   std::cout << "\nmean trained perf = (" << best_perf_mean << ", " << perf_stdev << ")\n";
   std::set<flexnnet::TrainingRecord> trecs = tr.get_records();
   std::cout << "\nsuccess rate = " << 100*success_rate << "%\n";

   const std::vector<flexnnet::TrainingRecordEntry> trace = trecs.begin()->training_set_trace;

   // best mean performance should be trained to error goal previously seen
   EXPECT_LE(best_perf_mean, ERROR_GOAL) << "Best training performance didn't reach expected goal.";

   // standard deviation of error across runs should be less than 10%
   EXPECT_LE(perf_stdev, 0.1 * best_perf_mean) << "Training run performance greater than expected.";

   // successful training rate should be greater than 90%
   EXPECT_GE(success_rate, 0.9) << "Successful training rate less than expected.";

/*
   std::cout << "perf of first entry = " << trecs.begin()->best_performance << "\n";
   for (auto it = trace.begin(); it != trace.end(); it++)
      std::cout << it->epoch << " " << it->performance << "\n";
*/

   std::cout << "\n eval after return\n";
   double tst_perf, tst_stdev;
   std::tie(tst_perf,tst_stdev) = eval.evaluate(nnet, tstset);
   std::cout << tst_perf << ", " << tst_stdev << "\n";

   // Performance on the test set should meet expected goal.
   EXPECT_LE(tst_perf, ERROR_GOAL) << "test set performance should match expected goal.";

   std::cout << "****** top best performances *****" << "\n";
   for (auto it = trecs.begin(); it != trecs.end(); it++)
      std::cout << it->best_performance << "\n";
}

TEST_F (SupervisedTrainerTestFixture, TanSigHiddenClassifierTrainingTest)
{
   /*
    * Generate the training data
    */
   const double MEAN_A = 0.4;
   const double STD_A = 0.4;

   const double MEAN_B = -0.5;
   const double STD_B = 0.5;

   const double ERROR_GOAL = 0.5;

   SimpleBinaryClassifierDataSet trnset, tstset;

   trnset.generate_samples(43, 0, MEAN_A, STD_A);
   trnset.generate_samples(51, 1, MEAN_B, STD_B);
   EXPECT_NO_THROW("Unexpected exception thrown while building training data set.");

   trnset.generate_samples(243, 0, MEAN_A, STD_A);
   trnset.generate_samples(251, 1, MEAN_B, STD_B);
   EXPECT_NO_THROW("Unexpected exception thrown while building training data set.");

   /*
    * Set up RMSE fitness and performance evaluator.
    */
   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;

   /*
    * Configure NN topology and base neural net
    */
   flexnnet::ValarrMap nninput;
   nninput["input"] = {1};
   flexnnet::NetworkTopology nntopo(nninput);
   nntopo.add_layer<flexnnet::TanSig>("output", 1, true);
   nntopo.add_layer<flexnnet::TanSig>("hidden", 2, false);
   nntopo.add_layer_connection("output", "hidden");
   nntopo.add_external_input_field("hidden", "input");
   EXPECT_NO_THROW("Unexpected exception thrown while building NetworkTopology.");

   flexnnet::BaseNeuralNet basenet(nntopo);
   EXPECT_NO_THROW("Unexpected exception thrown while defining BaseNeuralNet.");

   /*
    * Define templatized neural network
    */
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       NeuralNet,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   EXPECT_NO_THROW("Unexpected exception thrown while defining NN<>.");

   /*
    * Define and configure trainer
    */
   flexnnet::SuperviseTrainingAlgo<flexnnet::ValarrayMap,
                                   flexnnet::ValarrayMap,
                                   NeuralNet,
                                   flexnnet::DataSet,
                                   flexnnet::Evaluator,
                                   flexnnet::RMSEFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer(nnet);

   trainer.set_saved_nnet_limit(20);
   trainer.set_batch_mode(0);
   trainer.set_training_runs(20);
   trainer.set_max_epochs(750);
   trainer.set_learning_rate(0.01);
   trainer.set_error_goal(ERROR_GOAL);
   trainer.set_report_frequency(1);
   //trainer.set_train_biases("output", false);

   EXPECT_NO_THROW("Unexpected exception thrown while configuring trainer.");

   flexnnet::NetworkWeights nw = nnet.get_weights();
   flexnnet::LayerWeights iow = nw.at("output");
   flexnnet::LayerWeights ihw = nw.at("hidden");

   trainer.train(trnset);
   EXPECT_NO_THROW("Unexpected exception thrown while training network.");

   nw = nnet.get_weights();
   flexnnet::LayerWeights tow = nw.at("output");
   flexnnet::LayerWeights thw = nw.at("hidden");

/*
   std::cout << prettyPrintArray("initial output weights", iow.const_weights_ref);

   std::cout << prettyPrintArray("trained output weights", tow.const_weights_ref);

   std::cout << prettyPrintArray("initial hidden weights", ihw.const_weights_ref);

   std::cout << prettyPrintArray("trained hidden weights", thw.const_weights_ref);
*/

   const flexnnet::TrainingReport& tr = trainer.get_training_report();

   std::cout << "\ntotal training runs = " << tr.total_training_runs() << "\n";

   double best_mean_perf, perf_stdev, success_rate;
   success_rate = tr.successful_training_rate();

   std::tie<double,double>(best_mean_perf, perf_stdev) = tr.performance_statistics();
   std::cout << "\nmean trained perf = (" << best_mean_perf << ", " << perf_stdev << ")\n";
   std::set<flexnnet::TrainingRecord> trecs = tr.get_records();
   std::cout << "\nsuccess rate = " << 100*success_rate << "%\n";

   // best mean performance should be trained to error goal previously seen
   EXPECT_LE(best_mean_perf, ERROR_GOAL) << "Best training performance didn't reach expected goal.";

   // standard deviation of error across runs should be less than 10%
   EXPECT_LE(perf_stdev, 0.1*best_mean_perf) << "Training run performance greater than expected.";

   // successful training rate should be greater than 90%
   EXPECT_GE(success_rate, 0.85) << "Successful training rate less than expected.";

   const std::vector<flexnnet::TrainingRecordEntry> trace = trecs.begin()->training_set_trace;

/*
   std::cout << "perf of first entry = " << trecs.begin()->best_performance << "\n";
   for (auto it = trace.begin(); it != trace.end(); it++)
      std::cout << it->epoch << ", " << it->performance << "\n";
*/

   std::ofstream of("binclassifier_trace.txt");
   for (auto it = trace.begin(); it != trace.end(); it++)
      of << it->epoch << ", " << it->performance << "\n";
   of.close();

/*

   std::cout << "perf of second entry = " << trecs.begin()->best_performance << "\n";

   std::set<flexnnet::TrainingRecord>::iterator trit = trecs.begin();
   trit++;
   const std::vector<flexnnet::TrainingRecordEntry> trace2 = trit->training_set_trace;
   for (auto it = trace2.begin(); it != trace2.end(); it++)
      std::cout << it->epoch << " " << it->performance << "\n";
*/

   std::cout << "****** top best performances *****" << "\n";
   for (auto it = trecs.begin(); it != trecs.end(); it++)
      std::cout << it->best_performance << " " << it->best_epoch << "\n";

   ValarrayMap nnout;


   std::ofstream cof("binclassifier_run.txt");
   trnset.randomize_order();
   std::cout << "\nactual classification\n";
   for (auto& it : trnset)
   {
      std::valarray<double> in = it.first.at("input");
      std::valarray<double> tgt = it.second.at("output");

      nnout = nnet.activate(it.first);
      std::valarray<double> outv = nnout.at("output");

      cof << in[0] << ", " << outv[0] << ", " << tgt[0] << "\n";
   }
   cof.close();

   double tst_perf, tst_stdev;
   std::tie(tst_perf,tst_stdev) = eval.evaluate(nnet, tstset);

   // Performance on the test set should meet expected goal.
   EXPECT_LE(tst_perf, ERROR_GOAL) << "test set performance should match expected goal.";
}



TEST_F (SupervisedTrainerTestFixture, LogSigHiddenClassifierTrainingTest)
{
   /*
    * Generate the training data
    */
   const double MEAN_A = 0.4;
   const double STD_A = 0.4;

   const double MEAN_B = -0.5;
   const double STD_B = 0.5;

   const double ERROR_GOAL = 0.5;

   SimpleBinaryClassifierDataSet trnset, tstset;

   trnset.generate_samples(43, 0, MEAN_A, STD_A);
   trnset.generate_samples(51, 1, MEAN_B, STD_B);
   EXPECT_NO_THROW("Unexpected exception thrown while building training data set.");

   trnset.generate_samples(243, 0, MEAN_A, STD_A);
   trnset.generate_samples(251, 1, MEAN_B, STD_B);
   EXPECT_NO_THROW("Unexpected exception thrown while building training data set.");

   /*
    * Set up RMSE fitness and performance evaluator.
    */
   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;

   /*
    * Configure NN topology and base neural net
    */
   flexnnet::ValarrMap nninput;
   nninput["input"] = {1};
   flexnnet::NetworkTopology nntopo(nninput);
   nntopo.add_layer<flexnnet::TanSig>("output", 1, true);
   nntopo.add_layer<flexnnet::LogSig>("hidden", 2, false);
   nntopo.add_layer_connection("output", "hidden");
   nntopo.add_external_input_field("hidden", "input");
   EXPECT_NO_THROW("Unexpected exception thrown while building NetworkTopology.");

   flexnnet::BaseNeuralNet basenet(nntopo);
   EXPECT_NO_THROW("Unexpected exception thrown while defining BaseNeuralNet.");

   /*
    * Define templatized neural network
    */
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       NeuralNet,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   EXPECT_NO_THROW("Unexpected exception thrown while defining NN<>.");

   /*
    * Define and configure trainer
    */
   flexnnet::SuperviseTrainingAlgo<flexnnet::ValarrayMap,
                                   flexnnet::ValarrayMap,
                                   NeuralNet,
                                   flexnnet::DataSet,
                                   flexnnet::Evaluator,
                                   flexnnet::RMSEFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer(nnet);

   trainer.set_saved_nnet_limit(20);
   trainer.set_batch_mode(0);
   trainer.set_training_runs(20);
   trainer.set_max_epochs(750);
   trainer.set_learning_rate(0.01);
   trainer.set_error_goal(ERROR_GOAL);
   trainer.set_report_frequency(1);
   //trainer.set_train_biases("output", false);

   EXPECT_NO_THROW("Unexpected exception thrown while configuring trainer.");

   flexnnet::NetworkWeights nw = nnet.get_weights();
   flexnnet::LayerWeights iow = nw.at("output");
   flexnnet::LayerWeights ihw = nw.at("hidden");

   trainer.train(trnset);
   EXPECT_NO_THROW("Unexpected exception thrown while training network.");

   nw = nnet.get_weights();
   flexnnet::LayerWeights tow = nw.at("output");
   flexnnet::LayerWeights thw = nw.at("hidden");

/*
   std::cout << prettyPrintArray("initial output weights", iow.const_weights_ref);

   std::cout << prettyPrintArray("trained output weights", tow.const_weights_ref);

   std::cout << prettyPrintArray("initial hidden weights", ihw.const_weights_ref);

   std::cout << prettyPrintArray("trained hidden weights", thw.const_weights_ref);
*/


   const flexnnet::TrainingReport& tr = trainer.get_training_report();

   std::cout << "\ntotal training runs = " << tr.total_training_runs() << "\n";

   double best_mean_perf, perf_stdev, success_rate;
   success_rate = tr.successful_training_rate();

   std::tie<double,double>(best_mean_perf, perf_stdev) = tr.performance_statistics();
   std::cout << "\nmean trained perf = (" << best_mean_perf << ", " << perf_stdev << ")\n";
   std::set<flexnnet::TrainingRecord> trecs = tr.get_records();
   std::cout << "\nsuccess rate = " << 100*success_rate << "%\n";

   // best mean performance should be trained to error goal previously seen
   EXPECT_LE(best_mean_perf, ERROR_GOAL) << "Best training performance didn't reach expected goal.";

   // standard deviation of error across runs should be less than 10%
   EXPECT_LE(perf_stdev, 0.1*best_mean_perf) << "Training run performance greater than expected.";

   // successful training rate should be greater than 90%
   EXPECT_GE(success_rate, 0.9) << "Successful training rate less than expected.";

   const std::vector<flexnnet::TrainingRecordEntry> trace = trecs.begin()->training_set_trace;

/*
   std::cout << "perf of first entry = " << trecs.begin()->best_performance << "\n";
   for (auto it = trace.begin(); it != trace.end(); it++)
      std::cout << it->epoch << ", " << it->performance << "\n";
*/

   std::ofstream of("binclassifier_trace.txt");
   for (auto it = trace.begin(); it != trace.end(); it++)
      of << it->epoch << ", " << it->performance << "\n";
   of.close();

/*

   std::cout << "perf of second entry = " << trecs.begin()->best_performance << "\n";

   std::set<flexnnet::TrainingRecord>::iterator trit = trecs.begin();
   trit++;
   const std::vector<flexnnet::TrainingRecordEntry> trace2 = trit->training_set_trace;
   for (auto it = trace2.begin(); it != trace2.end(); it++)
      std::cout << it->epoch << " " << it->performance << "\n";
*/

   std::cout << "****** top best performances *****" << "\n";
   for (auto it = trecs.begin(); it != trecs.end(); it++)
      std::cout << it->best_performance << " " << it->best_epoch << "\n";

   ValarrayMap nnout;


   std::ofstream cof("binclassifier_run.txt");
   trnset.randomize_order();
   std::cout << "\nactual classification\n";
   for (auto& it : trnset)
   {
      std::valarray<double> in = it.first.at("input");
      std::valarray<double> tgt = it.second.at("output");

      nnout = nnet.activate(it.first);
      std::valarray<double> outv = nnout.at("output");

      cof << in[0] << ", " << outv[0] << ", " << tgt[0] << "\n";
   }
   cof.close();

   double tst_perf, tst_stdev;
   std::tie(tst_perf,tst_stdev) =eval.evaluate(nnet, tstset);

   // Performance on the test set should meet expected goal.
   EXPECT_LE(tst_perf, ERROR_GOAL) << "test set performance should match expected goal.";
}

TEST_F (SupervisedTrainerTestFixture, LinHiddenClassifierTrainingTest)
{
   /*
    * Generate the training data
    */
   const double MEAN_A = 0.4;
   const double STD_A = 0.4;

   const double MEAN_B = -0.5;
   const double STD_B = 0.5;

   const double ERROR_GOAL = 0.5;

   SimpleBinaryClassifierDataSet trnset, tstset;

   trnset.generate_samples(43, 0, MEAN_A, STD_A);
   trnset.generate_samples(51, 1, MEAN_B, STD_B);
   EXPECT_NO_THROW("Unexpected exception thrown while building training data set.");

   trnset.generate_samples(243, 0, MEAN_A, STD_A);
   trnset.generate_samples(251, 1, MEAN_B, STD_B);
   EXPECT_NO_THROW("Unexpected exception thrown while building training data set.");

   /*
    * Set up RMSE fitness and performance evaluator.
    */
   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;

   /*
    * Configure NN topology and base neural net
    */
   flexnnet::ValarrMap nninput;
   nninput["input"] = {1};
   flexnnet::NetworkTopology nntopo(nninput);
   nntopo.add_layer<flexnnet::TanSig>("output", 1, true);
   nntopo.add_layer<flexnnet::PureLin>("hidden", 3, false);
   nntopo.add_layer_connection("output", "hidden");
   nntopo.add_external_input_field("hidden", "input");
   EXPECT_NO_THROW("Unexpected exception thrown while building NetworkTopology.");

   flexnnet::BaseNeuralNet basenet(nntopo);
   EXPECT_NO_THROW("Unexpected exception thrown while defining BaseNeuralNet.");

   /*
    * Define templatized neural network
    */
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       NeuralNet,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   EXPECT_NO_THROW("Unexpected exception thrown while defining NN<>.");

   /*
    * Define and configure trainer
    */
   flexnnet::SuperviseTrainingAlgo<flexnnet::ValarrayMap,
                                   flexnnet::ValarrayMap,
                                   NeuralNet,
                                   flexnnet::DataSet,
                                   flexnnet::Evaluator,
                                   flexnnet::RMSEFitnessFunc,
                                   flexnnet::DeltaBarDeltaLearningRate> trainer(nnet);

   trainer.set_batch_mode(0);
   trainer.set_training_runs(20);
   trainer.set_max_epochs(750);
   trainer.set_learning_rate(0.01);
   trainer.set_error_goal(ERROR_GOAL);
   trainer.set_report_frequency(1);
   trainer.set_saved_nnet_limit(20);
   //trainer.set_train_biases("hidden", false);

   EXPECT_NO_THROW("Unexpected exception thrown while configuring trainer.");

   flexnnet::NetworkWeights nw = nnet.get_weights();
   flexnnet::LayerWeights iow = nw.at("output");
   flexnnet::LayerWeights ihw = nw.at("hidden");

   trainer.train(trnset);
   EXPECT_NO_THROW("Unexpected exception thrown while training network.");

   nw = nnet.get_weights();
   flexnnet::LayerWeights tow = nw.at("output");
   flexnnet::LayerWeights thw = nw.at("hidden");

/*
   std::cout << prettyPrintArray("initial output weights", iow.const_weights_ref);

   std::cout << prettyPrintArray("trained output weights", tow.const_weights_ref);

   std::cout << prettyPrintArray("initial hidden weights", ihw.const_weights_ref);

   std::cout << prettyPrintArray("trained hidden weights", thw.const_weights_ref);
*/


   const flexnnet::TrainingReport& tr = trainer.get_training_report();

   std::cout << "\ntotal training runs = " << tr.total_training_runs() << "\n";

   double best_mean_perf, perf_stdev, success_rate;
   success_rate = tr.successful_training_rate();

   std::tie<double,double>(best_mean_perf, perf_stdev) = tr.performance_statistics();
   std::cout << "\nmean trained perf = (" << best_mean_perf << ", " << perf_stdev << ")\n";
   std::set<flexnnet::TrainingRecord> trecs = tr.get_records();
   std::cout << "\nsuccess rate = " << 100*success_rate << "%\n";

   // best mean performance should be trained to error goal previously seen
   EXPECT_LE(best_mean_perf, ERROR_GOAL) << "Best training performance didn't reach expected goal.";

   // standard deviation of error across runs should be less than 10%
   EXPECT_LE(perf_stdev, 0.1*best_mean_perf) << "Training run performance greater than expected.";

   // successful training rate should be greater than 90%
   EXPECT_GE(success_rate, 0.9) << "Successful training rate less than expected.";

   const std::vector<flexnnet::TrainingRecordEntry> trace = trecs.begin()->training_set_trace;


/*   std::cout << "perf of first entry = " << trecs.begin()->best_performance << "\n";
   for (auto it = trace.begin(); it != trace.end(); it++)
      std::cout << it->epoch << ", " << it->performance << "\n";*/


   std::ofstream of("binclassifier_trace.txt");
   for (auto it = trace.begin(); it != trace.end(); it++)
      of << it->epoch << ", " << it->performance << "\n";
   of.close();

/*

   std::cout << "perf of second entry = " << trecs.begin()->best_performance << "\n";

   std::set<flexnnet::TrainingRecord>::iterator trit = trecs.begin();
   trit++;
   const std::vector<flexnnet::TrainingRecordEntry> trace2 = trit->training_set_trace;
   for (auto it = trace2.begin(); it != trace2.end(); it++)
      std::cout << it->epoch << " " << it->performance << "\n";
*/

   std::cout << "****** top best performances *****" << "\n";
   for (auto it = trecs.begin(); it != trecs.end(); it++)
      std::cout << it->best_performance << " " << it->best_epoch << "\n";

   ValarrayMap nnout;

   std::ofstream cof("binclassifier_run.txt");
   trnset.randomize_order();
   std::cout << "\nactual classification\n";
   for (auto& it : trnset)
   {
      std::valarray<double> in = it.first.at("input");
      std::valarray<double> tgt = it.second.at("output");

      nnout = nnet.activate(it.first);
      std::valarray<double> outv = nnout.at("output");

      cof << in[0] << ", " << outv[0] << ", " << tgt[0] << "\n";
   }
   cof.close();

   double tst_perf, tst_stdev;
   std::tie(tst_perf,tst_stdev) =eval.evaluate(nnet, tstset);

   std::cout << "\n--- test set performance ---\n";
   std::cout << tst_perf << ", " << tst_stdev << "\n";

   // Performance on the test set should meet expected goal.
   EXPECT_LE(tst_perf, ERROR_GOAL) << "test set performance should match expected goal.";
}

#endif //_CLASSIFIERTRAININGTESTS_H_
