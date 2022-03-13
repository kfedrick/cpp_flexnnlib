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
#include "Reinforcement.h"

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
   DataSet<FeatureSetImpl<std::tuple<RawFeature<1>>>, FeatureSetImpl<std::tuple<RawFeature<1>>>, flexnnet::ExemplarSeries> trnset;

   flexnnet::RMSEFitnessFunc<flexnnet::FeatureSetImpl<std::tuple<RawFeature<1>>>> rmse_fit;

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr =
      std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "output", TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 1);


   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   BaseNeuralNet newbasennet(topo);
   NeuralNet<flexnnet::FeatureSetImpl<std::tuple<RawFeature<1>>>, flexnnet::FeatureSetImpl<std::tuple<RawFeature<1>>>> newnnet(newbasennet);

   flexnnet::DeepRLAlgo<flexnnet::FeatureSetImpl<std::tuple<RawFeature<1>>>,
                        flexnnet::FeatureSetImpl<std::tuple<RawFeature<1>>>,
                        NeuralNet,
                        flexnnet::DataSet,
                        flexnnet::TDFinalFitnessFunc,
                        flexnnet::ConstantLearningRate> trainer(newnnet);
}

/*

TEST_F (SupervisedTrainerTestFixture, BounderRandomWalkTest)
{
   flexnnet::RMSEFitnessFunc<FeatureSet<std::tuple<RawFeature<1>>>> td_fit;

   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr =
      std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "output", TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("F0", 9+2);

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   BaseNeuralNet newbasennet(topo);
   NeuralNet<FeatureSet<std::tuple<RawFeature<9+2>>>, FeatureSet<std::tuple<RawFeature<1>>>> newnnet(newbasennet);

   flexnnet::DeepRLAlgo<FeatureSet<std::tuple<RawFeature<9+2>>>,
                        FeatureSet<std::tuple<RawFeature<1>>>,
                        NeuralNet,
                        DataSet,
                        flexnnet::TDFinalFitnessFunc,
                        flexnnet::ConstantLearningRate> trainer(newnnet);

   BoundedRandomWalkDataSet<9> trnset;

   ExemplarSeries<FeatureSet<std::tuple<RawFeature<9+2>>>, FeatureSet<std::tuple<RawFeature<1>>>> eseries;
   Exemplar<FeatureSet<std::tuple<RawFeature<9+2>>>, FeatureSet<std::tuple<RawFeature<1>>>> exemplar;

   //trnset.generate_final_cost_samples(1000, 7);
   trnset.generate_final_cost_samples(20, 9);

   std::cout << "# of series = " << trnset.size() << "\n" << std::flush;

   trainer.set_training_runs(1);
   trainer.set_lambda(0.3);
   trainer.set_max_epochs(10);
   trainer.set_batch_mode(3);
   trainer.set_td_mode(flexnnet::TDTrainerConfig::FINAL_COST);

   newnnet.initialize_weights();

   flexnnet::LayerWeights iow = newnnet.get_weights("output");

   trainer.train(trnset);
   std::cout << "DeepRLTests.BounderRandomWalkTest DONE\n" << std::flush;

   flexnnet::LayerWeights tow = newnnet.get_weights("output");

   std::cout << prettyPrintArray("initial output weights", iow.const_weights_ref);

   std::cout << prettyPrintArray("trained output weights", tow.const_weights_ref);

   const flexnnet::TrainingReport& tr = trainer.get_training_report();
   std::set<flexnnet::TrainingRecord> trecs = tr.get_records();
   std::cout << "\ntotal training runs = " << tr.total_training_runs() << "\n";

   std::cout << "\nperf of first entry = " << trecs.begin()->best_epoch << " " << trecs.begin()->best_performance << "\n";
   const std::vector<flexnnet::TrainingRecordEntry> trace = trecs.begin()->training_set_trace;
   for (auto it = trace.begin(); it != trace.end(); it++)
      std::cout << it->epoch << " " << it->performance << "\n";

   double best_perf_mean, perf_stdev;
   double success_rate;

   success_rate = tr.successful_training_rate();
   std::tie<double,double>(best_perf_mean, perf_stdev) = tr.performance_statistics();
   std::cout << "\nmean trained perf = (" << best_perf_mean << ", " << perf_stdev << ")\n";
   std::cout << "\nsuccess rate = " << 100*success_rate << "%\n";

   std::cout << "****** top best performances *****" << "\n";
   for (auto it = trecs.begin(); it != trecs.end(); it++)
      std::cout << it->best_performance << " at epoch " << it->best_epoch << "\n";

   int count = 0;
   std::valarray<double> invec(9+2);
   FeatureSet<std::tuple<RawFeature<9+2>>> infeature;

   for (int pos=1; pos<=9; pos++)
   {
      invec = -1.0;
      invec[pos] = 1.0;

      std::get<0>(infeature.get_features()).decode(invec);

      RawFeature<9+2> if0 = std::get<0>(infeature.get_features());
      std::valarray<double> ifv = if0.get_encoding();
      std::cout << prettyPrintVector("inputv", ifv);

      FeatureSet<std::tuple<RawFeature<1>>> nnout = newnnet.activate(infeature);
      RawFeature<1> of0 = std::get<0>(nnout.get_features());
      std::valarray<double> ofv = of0.get_encoding();

      std::cout << prettyPrintVector("nnout", ofv);
   }
}
*/


TEST_F (SupervisedTrainerTestFixture, C2GBoundedRandomWalkTest)
{
/*   FeatureSet<std::tuple<RawFeature<9+2>>> tstmap;
   tstmap["output"] = {};
   const std::valarray<double>& item = tstmap.value();
   std::cout << "OK this seems to work.\n" << std::flush;*/

   flexnnet::RMSEFitnessFunc<FeatureSetImpl<std::tuple<RawFeature<1>>>> rmse_fit;

   std::shared_ptr<NetworkLayerImpl<PureLin>> ol_ptr =
      std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(1, "output", PureLin::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("F0", 9+2);

   NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   BaseNeuralNet newbasennet(topo);
   NeuralNet<FeatureSetImpl<std::tuple<RawFeature<9+2>>>, FeatureSetImpl<std::tuple<RawFeature<1>>>> newnnet(newbasennet);

   flexnnet::DeepRLAlgo<FeatureSetImpl<std::tuple<RawFeature<9+2>>>,
                        FeatureSetImpl<std::tuple<RawFeature<1>>>,
                        NeuralNet,
                        DataSet,
                        flexnnet::TDCostToGoFitnessFunc,
                        flexnnet::ConstantLearningRate> trainer(newnnet);

   BoundedRandomWalkDataSet<9> trnset;

   ExemplarSeries<FeatureSetImpl<std::tuple<RawFeature<9+2>>>, FeatureSetImpl<std::tuple<RawFeature<1>>>> eseries;
   Exemplar<FeatureSetImpl<std::tuple<RawFeature<9+2>>>, FeatureSetImpl<std::tuple<RawFeature<1>>>> exemplar;

   trnset.generate_cost_to_go_samples(100, 9);

   std::cout << "# of series = " << trnset.size() << "\n" << std::flush;

   trainer.set_training_runs(1);
   //trainer.set_batch_mode(3);
   trainer.set_max_epochs(25);
   trainer.set_gamma(0.98);
   trainer.set_lambda(0.3);
   trainer.set_learning_rate(0.001);
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
      //std::cout << it->epoch << " " << it->performance << "\n";
      std::cout << it->performance << "\n";

   double best_perf_mean, perf_stdev;
   double success_rate;

   success_rate = tr.successful_training_rate();
   std::tie<double,double>(best_perf_mean, perf_stdev) = tr.performance_statistics();
   std::cout << "\nmean trained perf = (" << best_perf_mean << ", " << perf_stdev << ")\n";
   std::cout << "\nsuccess rate = " << 100*success_rate << "%\n";

   std::cout << "****** top best performances *****" << "\n";
   for (auto it = trecs.begin(); it != trecs.end(); it++)
      std::cout << it->best_performance << " at epoch " << it->best_epoch << "\n";

   int count = 0;
   std::valarray<double> invec(9+2);
   FeatureSetImpl<std::tuple<RawFeature<9+2>>> infeature;

   for (int pos=1; pos<=9; pos++)
   {
      invec = -1.0;
      invec[pos] = 1.0;

      std::get<0>(infeature.get_features()).decode(invec);

      RawFeature<9+2> if0 = std::get<0>(infeature.get_features());
      std::valarray<double> ifv = if0.get_encoding();
      std::cout << prettyPrintVector("inputv", ifv);

      FeatureSetImpl<std::tuple<RawFeature<1>>> nnout = newnnet.activate(infeature);
      RawFeature<1> of0 = std::get<0>(nnout.get_features());
      std::valarray<double> ofv = of0.get_encoding();

      std::cout << prettyPrintVector("nnout", ofv);
   }
   /*
   int count = 0;
   for (auto& aseries2 : trnset)
   {
      if (count++ > 3)
         break;

      for (auto& x : aseries2)
      {
         std::cout << prettyPrintVector("inputv", x.first.at("input"));
         FeatureSet<std::tuple<RawFeature<3>>> nnout = newnnet.activate(x.first);
         std::cout << prettyPrintVector("nnout", nnout.at("output"));
      }
      std::cout << "\n************************************\n";
   }
   */
}


#endif //_CLASSIFIERTRAININGTESTS_H_
