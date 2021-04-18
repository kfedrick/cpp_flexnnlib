//
// Created by kfedrick on 3/29/21.
//

#ifndef _CLASSIFIERTRAININGTESTS_H_
#define _CLASSIFIERTRAININGTESTS_H_

#include "gtest/gtest.h"

#include "SupervisedTrainerTestFixture.h"

#include "flexnnet.h"
#include "Evaluator.h"
#include "DataSet.h"
#include "RMSEFitnessFunc.h"
#include "Evaluator.h"
#include <ValarrayMap.h>
#include <fstream>
#include <CommonTestFixtureFunctions.h>
#include <PureLin.h>
#include <DeltaBarDeltaLearningRate.h>
#include <TanSig.h>
#include <SoftMax.h>
#include <LogSig.h>
#include <RadBas.h>
#include <NeuralNetTopology.h>
#include <NetworkLayerImpl.h>
#include <BaseNeuralNet.h>
#include <NeuralNet.h>
#include <SuperviseTrainingAlgo.h>
#include <ConstantLearningRate.h>

#include "SimpleBinaryClassifierDataSet.h"

using flexnnet::PureLin;
using flexnnet::TanSig;
using flexnnet::LogSig;

using flexnnet::NeuralNetTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::NetworkLayerImpl;
using flexnnet::NeuralNet;


TEST_F (SupervisedTrainerTestFixture, NewSingleLinBinClassifierTrainingTest)
{
   /*
    * Generate the training data
    */
   const double MEAN_A = 1.0;
   const double STD_A = 0.01;

   const double MEAN_B = -1.0;
   const double STD_B = 0.025;

   const double ERROR_GOAL = 0.15;

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

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr = std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "output", TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 1);
   /*
    * Configure NN topology and base neural net
    */
   flexnnet::ValarrMap nninput;
   nninput["input"] = {1};

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   EXPECT_NO_THROW("Unexpected exception thrown while building NeuralNetTopology.");

   BaseNeuralNet newbasennet(topo);
   EXPECT_NO_THROW("Unexpected exception thrown while defining BaseNeuralNet.");

   /*
    * Define templatized neural network
    */
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> newnnet(newbasennet);
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
                                   flexnnet::ConstantLearningRate> trainer(newnnet);

   trainer.set_saved_nnet_limit(20);
   trainer.set_batch_mode(0);
   trainer.set_training_runs(10);
   trainer.set_max_epochs(100);
   trainer.set_learning_rate(0.001);
   trainer.set_error_goal(ERROR_GOAL);
   //trainer.set_train_biases("output", false);

   EXPECT_NO_THROW("Unexpected exception thrown while configuring trainer.");

   flexnnet::LayerWeights iow = newnnet.get_weights("output");

   trainer.train(trnset);
   EXPECT_NO_THROW("Unexpected exception thrown while training network.");

   flexnnet::LayerWeights tow = newnnet.get_weights("output");

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
   std::tie(tst_perf,tst_stdev) = eval.evaluate(newnnet, tstset);
   std::cout << tst_perf << ", " << tst_stdev << "\n";

   // Performance on the test set should meet expected goal.
   EXPECT_LE(tst_perf, ERROR_GOAL) << "test set performance should match expected goal.";

   std::cout << "****** top best performances *****" << "\n";
   for (auto it = trecs.begin(); it != trecs.end(); it++)
      std::cout << it->best_performance << "\n";
}


TEST_F (SupervisedTrainerTestFixture, NewTanSigHiddenClassifierTrainingTest)
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

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr = std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "output", TanSig::DEFAULT_PARAMS, true));
   std::shared_ptr<NetworkLayerImpl<TanSig>> hl_ptr = std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(7, "hidden", TanSig::DEFAULT_PARAMS, false));

   hl_ptr->add_external_input_field("input", 1);
   ol_ptr->add_connection("activation", hl_ptr, flexnnet::LayerConnRecord::Forward);
   hl_ptr->add_connection("backprop", ol_ptr, flexnnet::LayerConnRecord::Forward);

   /*
    * Configure NN topology and base neural net
    */
   flexnnet::ValarrMap nninput;
   nninput["input"] = {1};

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_layers[hl_ptr->name()] = hl_ptr;

   topo.network_output_layers.push_back(ol_ptr);

   topo.ordered_layers.push_back(hl_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   EXPECT_NO_THROW("Unexpected exception thrown while building NeuralNetTopology.");

   BaseNeuralNet newbasennet(topo);
   EXPECT_NO_THROW("Unexpected exception thrown while defining BaseNeuralNet.");

   /*
    * Define templatized neural network
    */
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> newnnet(newbasennet);
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
                                   flexnnet::ConstantLearningRate> trainer(newnnet);

   trainer.set_saved_nnet_limit(20);
   trainer.set_batch_mode(0);
   trainer.set_training_runs(1);
   trainer.set_max_epochs(200);
   trainer.set_learning_rate(0.001);
   trainer.set_error_goal(ERROR_GOAL);
   trainer.set_report_frequency(1);
   //trainer.set_train_biases("output", false);

   EXPECT_NO_THROW("Unexpected exception thrown while configuring trainer.");

   newnnet.initialize_weights();

   flexnnet::LayerWeights iow = newnnet.get_weights("output");
   flexnnet::LayerWeights ihw = newnnet.get_weights("hidden");

   trainer.train(trnset);
   EXPECT_NO_THROW("Unexpected exception thrown while training network.");

   flexnnet::LayerWeights tow = newnnet.get_weights("output");
   flexnnet::LayerWeights thw = newnnet.get_weights("hidden");

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

/*   std::ofstream of("binclassifier_trace.txt");
   for (auto it = trace.begin(); it != trace.end(); it++)
      of << it->epoch << ", " << it->performance << "\n";
   of.close();*/

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

/*   ValarrayMap nnout;
   std::ofstream cof("binclassifier_run.txt");
   trnset.randomize_order();
   std::cout << "\nactual classification\n";
   for (auto& it : trnset)
   {
      std::valarray<double> in = it.first.at("input");
      std::valarray<double> tgt = it.second.at("output");

      nnout = newnnet.activate(it.first);
      std::valarray<double> outv = nnout.at("output");

      cof << in[0] << ", " << outv[0] << ", " << tgt[0] << "\n";
   }
   cof.close();*/

   double tst_perf, tst_stdev;
   std::tie(tst_perf,tst_stdev) = eval.evaluate(newnnet, tstset);

   // Performance on the test set should meet expected goal.
   EXPECT_LE(tst_perf, ERROR_GOAL) << "test set performance should match expected goal.";
}

TEST_F (SupervisedTrainerTestFixture, NewLogSigHiddenClassifierTrainingTest)
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

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr = std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "output", TanSig::DEFAULT_PARAMS, true));
   std::shared_ptr<NetworkLayerImpl<LogSig>> hl_ptr = std::make_shared<NetworkLayerImpl<LogSig>>(NetworkLayerImpl<LogSig>(5, "hidden", LogSig::DEFAULT_PARAMS, false));

   hl_ptr->add_external_input_field("input", 1);
   ol_ptr->add_connection("activation", hl_ptr, flexnnet::LayerConnRecord::Forward);
   hl_ptr->add_connection("backprop", ol_ptr, flexnnet::LayerConnRecord::Forward);

   /*
    * Configure NN topology and base neural net
    */
   //flexnnet::ValarrMap nninput;
   //nninput["input"] = {1};

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_layers[hl_ptr->name()] = hl_ptr;

   topo.network_output_layers.push_back(ol_ptr);

   topo.ordered_layers.push_back(hl_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   EXPECT_NO_THROW("Unexpected exception thrown while building NeuralNetTopology.");

   BaseNeuralNet newbasennet(topo);
   EXPECT_NO_THROW("Unexpected exception thrown while defining BaseNeuralNet.");

   /*
    * Define templatized neural network
    */
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> newnnet(newbasennet);
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
                                   flexnnet::ConstantLearningRate> trainer(newnnet);

   trainer.set_saved_nnet_limit(20);
   trainer.set_batch_mode(0);
   trainer.set_training_runs(20);
   trainer.set_max_epochs(200);
   trainer.set_learning_rate(0.001);
   trainer.set_error_goal(ERROR_GOAL);
   trainer.set_report_frequency(1);
   //trainer.set_train_biases("output", false);

   EXPECT_NO_THROW("Unexpected exception thrown while configuring trainer.");

   flexnnet::LayerWeights iow = newnnet.get_weights("output");
   flexnnet::LayerWeights ihw = newnnet.get_weights("hidden");

   trainer.train(trnset);
   EXPECT_NO_THROW("Unexpected exception thrown while training network.");

   flexnnet::LayerWeights tow = newnnet.get_weights("output");
   flexnnet::LayerWeights thw = newnnet.get_weights("hidden");

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


   std::ofstream of("binclassifier_trace.txt");
   for (auto it = trace.begin(); it != trace.end(); it++)
      of << it->epoch << ", " << it->performance << "\n";
   of.close();
*/

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

/*

   std::ofstream cof("binclassifier_run.txt");
   trnset.randomize_order();
   std::cout << "\nactual classification\n";
   for (auto& it : trnset)
   {
      std::valarray<double> in = it.first.at("input");
      std::valarray<double> tgt = it.second.at("output");

      nnout = newnnet.activate(it.first);
      std::valarray<double> outv = nnout.at("output");

      cof << in[0] << ", " << outv[0] << ", " << tgt[0] << "\n";
   }
   cof.close();
*/

   double tst_perf, tst_stdev;
   std::tie(tst_perf,tst_stdev) = eval.evaluate(newnnet, tstset);

   // Performance on the test set should meet expected goal.
   EXPECT_LE(tst_perf, ERROR_GOAL) << "test set performance should match expected goal.";
}


TEST_F (SupervisedTrainerTestFixture, NewPureLinHiddenClassifierTrainingTest)
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

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr = std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "output", TanSig::DEFAULT_PARAMS, true));
   std::shared_ptr<NetworkLayerImpl<PureLin>> hl_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(3, "hidden", PureLin::DEFAULT_PARAMS, false));

   hl_ptr->add_external_input_field("input", 1);
   ol_ptr->add_connection("activation", hl_ptr, flexnnet::LayerConnRecord::Forward);
   hl_ptr->add_connection("backprop", ol_ptr, flexnnet::LayerConnRecord::Forward);

   /*
    * Configure NN topology and base neural net
    */
   flexnnet::ValarrMap nninput;
   nninput["input"] = {1};

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_layers[hl_ptr->name()] = hl_ptr;

   topo.network_output_layers.push_back(ol_ptr);

   topo.ordered_layers.push_back(hl_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   EXPECT_NO_THROW("Unexpected exception thrown while building NeuralNetTopology.");

   BaseNeuralNet newbasennet(topo);
   EXPECT_NO_THROW("Unexpected exception thrown while defining BaseNeuralNet.");

   /*
    * Define templatized neural network
    */
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> newnnet(newbasennet);
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
                                   flexnnet::ConstantLearningRate> trainer(newnnet);

   trainer.set_saved_nnet_limit(20);
   trainer.set_batch_mode(0);
   trainer.set_training_runs(20);
   trainer.set_max_epochs(200);
   trainer.set_learning_rate(0.001);
   trainer.set_error_goal(ERROR_GOAL);
   trainer.set_report_frequency(1);
   //trainer.set_train_biases("output", false);

   EXPECT_NO_THROW("Unexpected exception thrown while configuring trainer.");

   flexnnet::LayerWeights iow = newnnet.get_weights("output");
   flexnnet::LayerWeights ihw = newnnet.get_weights("hidden");

   trainer.train(trnset);
   EXPECT_NO_THROW("Unexpected exception thrown while training network.");

   flexnnet::LayerWeights tow = newnnet.get_weights("output");
   flexnnet::LayerWeights thw = newnnet.get_weights("hidden");

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

/*   std::ofstream of("binclassifier_trace.txt");
   for (auto it = trace.begin(); it != trace.end(); it++)
      of << it->epoch << ", " << it->performance << "\n";
   of.close();*/

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

/*

   std::ofstream cof("binclassifier_run.txt");
   trnset.randomize_order();
   std::cout << "\nactual classification\n";
   for (auto& it : trnset)
   {
      std::valarray<double> in = it.first.at("input");
      std::valarray<double> tgt = it.second.at("output");

      nnout = newnnet.activate(it.first);
      std::valarray<double> outv = nnout.at("output");

      cof << in[0] << ", " << outv[0] << ", " << tgt[0] << "\n";
   }
   cof.close();
*/

   double tst_perf, tst_stdev;
   std::tie(tst_perf,tst_stdev) = eval.evaluate(newnnet, tstset);

   // Performance on the test set should meet expected goal.
   EXPECT_LE(tst_perf, ERROR_GOAL) << "test set performance should match expected goal.";
}


#endif //_CLASSIFIERTRAININGTESTS_H_
