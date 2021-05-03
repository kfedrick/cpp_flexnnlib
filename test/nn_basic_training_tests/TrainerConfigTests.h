//
// Created by kfedrick on 10/4/19.
//

#ifndef _TRAINERCONFIGTESTS_H_
#define _TRAINERCONFIGTESTS_H_

#include "gtest/gtest.h"

#include "SupervisedTrainerTestFixture.h"

#include "flexnnet.h"
#include "Evaluator.h"
#include "NeuralNetTopology.h"
#include "NeuralNet.h"
#include "DataSet.h"
#include "RMSEFitnessFunc.h"
#include "Evaluator.h"
#include <ValarrayMap.h>
#include <fstream>
#include <CommonTestFixtureFunctions.h>
#include <SupervisedTrainingAlgo.h>
#include <ConstantLearningRate.h>
#include "MockNN.h"

TEST_F (SupervisedTrainerTestFixture, BasicConstructor)
{
   std::cout << "***** Test Trainer Constructor\n" << std::flush;

   flexnnet::DataSet<flexnnet::ValarrayMap, flexnnet::ValarrayMap, Exemplar> dataset;
   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;
   flexnnet::BaseNeuralNet basenet;
   MockNN<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       MockNN,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   flexnnet::SupervisedTrainingAlgo<flexnnet::ValarrayMap,
                                    flexnnet::ValarrayMap,
                                    Exemplar,
                                    MockNN,
                                    flexnnet::DataSet,
                                    flexnnet::Evaluator,
                                    flexnnet::RMSEFitnessFunc,
                                    flexnnet::ConstantLearningRate> trainer(nnet);

   EXPECT_NO_THROW("Unexpected exception thrown.");
}

TEST_F (SupervisedTrainerTestFixture, BasicConfigTest)
{
   std::cout << "***** Test Basic Training Config Setters\n" << std::flush;

   flexnnet::DataSet<flexnnet::ValarrayMap, flexnnet::ValarrayMap, Exemplar> dataset;
   flexnnet::RMSEFitnessFunc<flexnnet::ValarrayMap> rmse_fit;
   flexnnet::BaseNeuralNet basenet;
   MockNN<flexnnet::ValarrayMap, flexnnet::ValarrayMap> nnet(basenet);
   flexnnet::Evaluator<flexnnet::ValarrayMap,
                       flexnnet::ValarrayMap,
                       MockNN,
                       flexnnet::DataSet,
                       flexnnet::RMSEFitnessFunc> eval;

   flexnnet::SupervisedTrainingAlgo<flexnnet::ValarrayMap,
                                    flexnnet::ValarrayMap,
                                    Exemplar,
                                    MockNN,
                                    flexnnet::DataSet,
                                    flexnnet::Evaluator,
                                    flexnnet::RMSEFitnessFunc,
                                    flexnnet::ConstantLearningRate> trainer(nnet);

   size_t BATCH = 5;
   trainer.set_batch_mode(BATCH);
   EXPECT_EQ(trainer.batch_mode(), BATCH)
               << "Batch mode = " << trainer.batch_mode() << " : expected " << BATCH
               << "\n";

   size_t RUNS = 10;
   trainer.set_training_runs(RUNS);
   EXPECT_EQ(trainer.training_runs(), RUNS)
               << "training runs = " << trainer.training_runs() << " : expected " << RUNS
               << "\n";

   size_t EPOCHS = 101;
   trainer.set_max_epochs(EPOCHS);
   EXPECT_EQ(trainer.max_epochs(), EPOCHS)
               << "max epochs = " << trainer.max_epochs() << " : expected " << EPOCHS
               << "\n";

   double EGOAL = 0.000001;
   trainer.set_error_goal(EGOAL);
   EXPECT_NEAR(trainer.error_goal(), EGOAL, 0.1e-9)
               << "error goal = " << trainer.error_goal() << " : expected " << EGOAL
               << "\n";

   EXPECT_NO_THROW("Unexpected exception thrown.");
}

#endif //_TRAINERCONFIGTESTS_H_
