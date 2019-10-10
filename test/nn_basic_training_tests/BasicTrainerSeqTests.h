//
// Created by kfedrick on 10/4/19.
//

#ifndef _BASICTRAINERSEQTESTS_H_
#define _BASICTRAINERSEQTESTS_H_

#include "gtest/gtest.h"

#include "NetworkLayer.h"
#include "BasicTrainerTests.h"

TYPED_TEST_P (BasicTrainerTests, BasicSequence)
{
   std::cout << "\nBasicTrainerTests::BasicSequence\n";

   TypeParam basic_trainer;

   basic_trainer.set_training_runs (2);
   basic_trainer.set_max_epochs (3);
   basic_trainer.train (*BasicTrainerTests<TypeParam>::nnet, BasicTrainerTests<TypeParam>::trnset);
}

#endif //_BASICTRAINERSEQTESTS_H_
