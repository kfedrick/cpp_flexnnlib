//
// Created by kfedrick on 3/19/21.
//

#ifndef _BASICCALLTREETESTS_H_
#define _BASICCALLTREETESTS_H_

#include "gtest/gtest.h"

#include "SupervisedTrainerTestFixture.h"

#include "flexnnet.h"
#include "NetworkTopology.h"
#include "NeuralNet.h"
#include "DataSet.h"
#include "RMSEFitnessFunc.h"
#include "Evaluator.h"
#include <ValarrayMap.h>
#include <fstream>
#include <CommonTestFixtureFunctions.h>
#include <SuperviseTrainingAlgo.h>
#include "MockNN.h"

TEST_F (SupervisedTrainerTestFixture, BasicCallTreeTest)
{
   std::cout << "***** Test Basic Supervised Training Function Call Tree\n" << std::flush;

   flexnnet::ValarrayMap tst1({{"output",{-1, 0, 0.5}}});
   flexnnet::ValarrayMap tgt1({{"output",{-1.03, 0.1, 0.59}}});
   flexnnet::ValarrayMap tst2({{"output",{-0.3, -1.3, 0.875}}});
   flexnnet::ValarrayMap tgt2({{"output",{-0.63, -0.1, 0.59}}});

   flexnnet::DataSet<flexnnet::ValarrayMap, flexnnet::ValarrayMap> trnset;
   trnset.push_back(std::pair<flexnnet::ValarrayMap, flexnnet::ValarrayMap>(tst1, tgt1));
   trnset.push_back(std::pair<flexnnet::ValarrayMap, flexnnet::ValarrayMap>(tst2, tgt2));

   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;
   flexnnet::BaseNeuralNet basenet(flexnnet::NetworkTopology({}));
   MockNN<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       MockNN,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   flexnnet::SuperviseTrainingAlgo<flexnnet::ValarrayMap,
                                   flexnnet::ValarrayMap,
                                   MockNN,
                                   flexnnet::DataSet,
                                   flexnnet::Evaluator,
                                   flexnnet::RMSEFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer(nnet);

   trainer.set_batch_mode(0);
   trainer.set_training_runs(3);
   trainer.set_max_epochs(7);

   trainer.train(trnset);

   EXPECT_NO_THROW("Unexpected exception thrown.");
}
#endif //_BASICCALLTREETESTS_H_
