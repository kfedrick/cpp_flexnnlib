//
// Created by kfedrick on 3/19/21.
//

#ifndef _BASICBACKPROPTESTS_H_
#define _BASICBACKPROPTESTS_H_

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
#include <ConstantLearningRate.h>
#include "MockNN.h"

TEST_F (SupervisedTrainerTestFixture, BackpropCallTreeTest)
{
   std::cout << "***** Test Basic Supervised Training Backprop Call Tree\n" << std::flush;

   flexnnet::ValarrayMap tst1({{"input",{-1, 0, 0.5}}});
   flexnnet::ValarrayMap tgt1({{"output",{-1.03, 0.1, 0.59}}});
   flexnnet::ValarrayMap tst2({{"input",{-0.3, -1.3, 0.875}}});
   flexnnet::ValarrayMap tgt2({{"output",{-0.63, -0.1, 0.59}}});

   flexnnet::DataSet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> trnset;
   trnset.push_back(std::pair<flexnnet::ValarrayMap, flexnnet::ValarrayMap>(tst1, tgt1));
   trnset.push_back(std::pair<flexnnet::ValarrayMap, flexnnet::ValarrayMap>(tst2, tgt2));

   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;

   flexnnet::ValarrMap nninput;
   nninput["input"] = {1,2,3};
   flexnnet::NetworkTopology nntopo(nninput);
   nntopo.add_layer<flexnnet::PureLin>("output", 3, true);
   nntopo.add_external_input_field("output", "input");

   flexnnet::BaseNeuralNet basenet(nntopo);


   MockNN<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       NeuralNet,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   flexnnet::SuperviseTrainingAlgo<flexnnet::ValarrayMap,
                                   flexnnet::ValarrayMap,
                                   NeuralNet,
                                   flexnnet::DataSet,
                                   flexnnet::Evaluator,
                                   flexnnet::RMSEFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer(nnet);

   trainer.set_batch_mode(0);
   trainer.set_training_runs(3);
   trainer.set_max_epochs(1);

   trainer.train(trnset);

   EXPECT_NO_THROW("Unexpected exception thrown.");

   std::cout << "*********************************************\n" << std::flush;
}

TEST_F (SupervisedTrainerTestFixture, BackpropTest)
{
   std::cout << "***** Test Basic Supervised Training Backprop Call Tree\n" << std::flush;

   flexnnet::ValarrayMap tst1({{"input",{-1, 0, 0.5}}});
   flexnnet::ValarrayMap tgt1({{"output",{-1.0, -1.0, 1.0}}});
   flexnnet::ValarrayMap tst2({{"input",{-0.3, -1.3, 0.875}}});
   flexnnet::ValarrayMap tgt2({{"output",{-1.0, 1.0, -1.0}}});

   flexnnet::DataSet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> trnset;
   trnset.push_back(std::pair<flexnnet::ValarrayMap, flexnnet::ValarrayMap>(tst1, tgt1));
   trnset.push_back(std::pair<flexnnet::ValarrayMap, flexnnet::ValarrayMap>(tst2, tgt2));

   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;

   flexnnet::ValarrMap nninput;
   nninput["input"] = {1,2,3};
   flexnnet::NetworkTopology nntopo(nninput);
   nntopo.add_layer<flexnnet::PureLin>("output", 3, true);
   nntopo.add_external_input_field("output", "input");

   flexnnet::BaseNeuralNet basenet(nntopo);


   MockNN<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       NeuralNet,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   flexnnet::SuperviseTrainingAlgo<flexnnet::ValarrayMap,
                                   flexnnet::ValarrayMap,
                                   NeuralNet,
                                   flexnnet::DataSet,
                                   flexnnet::Evaluator,
                                   flexnnet::RMSEFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer(nnet);

   trainer.set_batch_mode(0);
   trainer.set_training_runs(1);
   trainer.set_max_epochs(500);
   trainer.set_learning_rate(0.025);

   flexnnet::NetworkWeights nw = nnet.get_weights();
   flexnnet::LayerWeights ow = nw.at("output");

   const flexnnet::Array2D<double> dnet_dw = nnet.get_layers()["output"]->dnet_dw();


   std::cout << prettyPrintArray("output weights", ow.const_weights_ref);
   trainer.train(trnset);

   std::cout << prettyPrintArray("dnet_dw", dnet_dw);

   nw = nnet.get_weights();
   ow = nw.at("output");

   std::cout << prettyPrintArray("trained output weights", ow.const_weights_ref);

   EXPECT_NO_THROW("Unexpected exception thrown.");
}

TEST_F (SupervisedTrainerTestFixture, BackpropTest2)
{
   std::cout << "***** Test Basic Supervised Training\n" << std::flush;

   flexnnet::ValarrayMap tst1({{"input",{-1, 0, 0.5}}});
   flexnnet::ValarrayMap tgt1({{"output",{-1.0, -1.0, 1.0}}});
   flexnnet::ValarrayMap tst2({{"input",{-0.3, -1.3, 0.875}}});
   flexnnet::ValarrayMap tgt2({{"output",{-1.0, 1.0, -1.0}}});

   flexnnet::DataSet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> trnset;
   trnset.push_back(std::pair<flexnnet::ValarrayMap, flexnnet::ValarrayMap>(tst1, tgt1));
   trnset.push_back(std::pair<flexnnet::ValarrayMap, flexnnet::ValarrayMap>(tst2, tgt2));

   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;

   flexnnet::ValarrMap nninput;
   nninput["input"] = {1,2,3};
   flexnnet::NetworkTopology nntopo(nninput);
   nntopo.add_layer<flexnnet::PureLin>("output", 3, true);
   nntopo.add_external_input_field("output", "input");

   flexnnet::BaseNeuralNet basenet(nntopo);


   MockNN<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       NeuralNet,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   flexnnet::SuperviseTrainingAlgo<flexnnet::ValarrayMap,
                                   flexnnet::ValarrayMap,
                                   NeuralNet,
                                   flexnnet::DataSet,
                                   flexnnet::Evaluator,
                                   flexnnet::RMSEFitnessFunc,
                                   flexnnet::DeltaBarDeltaLearningRate> trainer(nnet);

   trainer.set_batch_mode(0);
   trainer.set_training_runs(3);
   trainer.set_max_epochs(100);
   trainer.set_learning_rate(0.025);
   trainer.set_error_goal(1e-15);
   //trainer.set_train_biases("output", false);

   flexnnet::NetworkWeights nw = nnet.get_weights();
   flexnnet::LayerWeights ow = nw.at("output");

   const flexnnet::Array2D<double> dnet_dw = nnet.get_layers()["output"]->dnet_dw();


   std::cout << prettyPrintArray("output weights", ow.const_weights_ref);
   trainer.train(trnset);

   std::cout << prettyPrintArray("dnet_dw", dnet_dw);

   nw = nnet.get_weights();
   ow = nw.at("output");

   std::cout << prettyPrintArray("trained output weights", ow.const_weights_ref);

   const flexnnet::TrainingReport& tr = trainer.get_training_report();

   std::cout << "\nno. of best networks = " << tr.get_records().size() << "\n";
   std::cout << "\ntotal training runs = " << tr.total_training_runs() << "\n";

   double pm, pv;
   std::tie<double,double>(pm, pv) = tr.performance_statistics();
   std::cout << "\nmean trained perf = (" << pm << ", " << pv << ")\n";
   std::set<flexnnet::TrainingRecord> trecs = tr.get_records();
   std::cout << "\nsuccess rate = " << tr.successful_training_rate() << "\n";
   std::cout << "perf of first entry = " << trecs.begin()->best_performance << "\n";

   const std::vector<flexnnet::TrainingRecordEntry> trace = trecs.begin()->training_set_trace;
   for (auto it = trace.begin(); it != trace.end(); it++)
      std::cout << it->epoch << " " << it->performance << "\n";

   std::cout << "****** top best performances *****" << "\n";
   for (auto it = trecs.begin(); it != trecs.end(); it++)
      std::cout << it->best_performance << "\n";

   EXPECT_NO_THROW("Unexpected exception thrown.");
}
#endif //_BASICBACKPROPTESTS_H_
