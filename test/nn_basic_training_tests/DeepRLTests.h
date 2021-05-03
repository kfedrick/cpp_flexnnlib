//
// Created by kfedrick on 3/29/21.
//

#ifndef _DEEPRLTRAININGTESTS_H_
#define _DEEPRLTRAININGTESTS_H_

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
#include <SupervisedTrainingAlgo.h>
#include <ConstantLearningRate.h>
#include <ExemplarSeries.h>
#include <DeepRLAlgo.h>
#include <CartesianCoord.h>
#include <TDFinalFitnessFunc.h>
#include <TDCostToGoFitnessFunc.h>

#include "SimpleBinaryClassifierDataSet.h"
#include "BoundedRandomWalkDataSet.h"

using flexnnet::PureLin;
using flexnnet::TanSig;
using flexnnet::LogSig;

using flexnnet::NeuralNetTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::NetworkLayerImpl;
using flexnnet::NeuralNet;
using flexnnet::CartesianCoord;
using flexnnet::ExemplarSeries;
using flexnnet::Exemplar;


TEST_F (SupervisedTrainerTestFixture, DeepRLAlgoConstructor)
{
   DataSet<ValarrayMap, ValarrayMap, flexnnet::ExemplarSeries> trnset;

   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr =
      std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "output", TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 1);


   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   BaseNeuralNet newbasennet(topo);
   NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> newnnet(newbasennet);

   flexnnet::DeepRLAlgo<flexnnet::ValarrayMap,
                        flexnnet::ValarrayMap,
                        NeuralNet,
                        flexnnet::DataSet,
                        flexnnet::TDFinalFitnessFunc,
                        flexnnet::ConstantLearningRate> trainer(newnnet);
}

TEST_F (SupervisedTrainerTestFixture, BounderRandomWalkTest)
{

   flexnnet::RMSEFitnessFunc<ValarrayMap> td_fit;

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr =
      std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "output", TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 9);

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   BaseNeuralNet newbasennet(topo);
   NeuralNet<ValarrayMap, ValarrayMap> newnnet(newbasennet);

   flexnnet::DeepRLAlgo<ValarrayMap,
                        ValarrayMap,
                        NeuralNet,
                        DataSet,
                        flexnnet::TDFinalFitnessFunc,
                        flexnnet::ConstantLearningRate> trainer(newnnet);

   BoundedRandomWalkDataSet trnset;

   ExemplarSeries<ValarrayMap, ValarrayMap> eseries;
   Exemplar<ValarrayMap, ValarrayMap> exemplar;

   trnset.generate_final_cost_samples(1000, 7);

   std::cout << "# of series = " << trnset.size() << "\n" << std::flush;

   trainer.set_training_runs(1);
   trainer.set_lambda(0.3);
   trainer.set_max_epochs(500);
   trainer.set_batch_mode(10);
   trainer.set_td_mode(flexnnet::TDTrainerConfig::FINAL_COST);

   newnnet.initialize_weights();

   flexnnet::LayerWeights iow = newnnet.get_weights("output");

   trainer.train(trnset);
   flexnnet::LayerWeights tow = newnnet.get_weights("output");

   std::cout << prettyPrintArray("initial output weights", iow.const_weights_ref);

   std::cout << prettyPrintArray("trained output weights", tow.const_weights_ref);

   const flexnnet::TrainingReport& tr = trainer.get_training_report();
   std::set<flexnnet::TrainingRecord> trecs = tr.get_records();


   std::cout << "\nperf of first entry = " << trecs.begin()->best_epoch << " " << trecs.begin()->best_performance << "\n";
   const std::vector<flexnnet::TrainingRecordEntry> trace = trecs.begin()->training_set_trace;
   for (auto it = trace.begin(); it != trace.end(); it++)
      std::cout << it->epoch << " " << it->performance << "\n";

   int count = 0;
   for (auto& aseries2 : trnset)
   {
      if (count++ > 3)
         break;

      for (auto& x : aseries2)
      {
         std::cout << prettyPrintVector("inputv", x.first.at("input"));
         ValarrayMap nnout = newnnet.activate(x.first);
         std::cout << prettyPrintVector("nnout", nnout.at("output"));
      }
      std::cout << "\n************************************\n";
   }
}

TEST_F (SupervisedTrainerTestFixture, C2GBoundedRandomWalkTest)
{
   ValarrayMap tstmap;
   tstmap["output"] = {};
   const std::valarray<double>& item = tstmap.value();
   std::cout << "OK this seems to work.\n" << std::flush;

   flexnnet::RMSEFitnessFunc<ValarrayMap> rmse_fit;

   std::shared_ptr<NetworkLayerImpl<PureLin>> ol_ptr =
      std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(1, "output", PureLin::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 9);

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   BaseNeuralNet newbasennet(topo);
   NeuralNet<ValarrayMap, ValarrayMap> newnnet(newbasennet);

   flexnnet::DeepRLAlgo<ValarrayMap,
                        ValarrayMap,
                        NeuralNet,
                        DataSet,
                        flexnnet::TDCostToGoFitnessFunc,
                        flexnnet::ConstantLearningRate> trainer(newnnet);

   BoundedRandomWalkDataSet trnset;

   ExemplarSeries<ValarrayMap, ValarrayMap> eseries;
   Exemplar<ValarrayMap, ValarrayMap> exemplar;

   trnset.generate_cost_to_go_samples(1000, 7);

   std::cout << "# of series = " << trnset.size() << "\n" << std::flush;

   trainer.set_training_runs(1);
   trainer.set_batch_mode(10);
   trainer.set_max_epochs(200);
   trainer.set_gamma(0.9);
   trainer.set_lambda(0.7);
   trainer.set_learning_rate(0.0001);
   trainer.set_td_mode(flexnnet::TDTrainerConfig::COST_TO_GO);

   newnnet.initialize_weights();

   flexnnet::LayerWeights iow = newnnet.get_weights("output");

   trainer.train(trnset);
   flexnnet::LayerWeights tow = newnnet.get_weights("output");

   std::cout << prettyPrintArray("initial output weights", iow.const_weights_ref);

   std::cout << prettyPrintArray("trained output weights", tow.const_weights_ref);

   const flexnnet::TrainingReport& tr = trainer.get_training_report();
   std::set<flexnnet::TrainingRecord> trecs = tr.get_records();

   std::cout << "\nperf of first entry = " << trecs.begin()->best_epoch << " " << trecs.begin()->best_performance << "\n";
   const std::vector<flexnnet::TrainingRecordEntry> trace = trecs.begin()->training_set_trace;
   for (auto it = trace.begin(); it != trace.end(); it++)
      std::cout << it->epoch << " " << it->performance << "\n";

   int count = 0;
   for (auto& aseries2 : trnset)
   {
      if (count++ > 3)
         break;

      for (auto& x : aseries2)
      {
         std::cout << prettyPrintVector("inputv", x.first.at("input"));
         ValarrayMap nnout = newnnet.activate(x.first);
         std::cout << prettyPrintVector("nnout", nnout.at("output"));
      }
      std::cout << "\n************************************\n";
   }
}

#endif //_CLASSIFIERTRAININGTESTS_H_
