//
// Created by kfedrick on 10/4/19.
//

#ifndef _NNEVALCONFIGTESTS_H_
#define _NNEVALCONFIGTESTS_H_

#include "gtest/gtest.h"

#include "BasicTrainerTests.h"

/*
TYPED_TEST_P (BasicTrainerTests, NNEvalConfig)
{
   std::cout << "\nBasicTrainerTests::NNEvalConfig\n";

   TypeParam basic_trainer;

   // ---   Set basic evaluator configuration variables

   // Check default sample size to draw for evaluation
   size_t DEFAULT_SAMPLE_SIZE = TypeParam::DEFAULT_SAMPLE_SIZE;
   ASSERT_EQ(basic_trainer.sample_size(), DEFAULT_SAMPLE_SIZE);

   // Set and test sample size
   size_t SAMPLE_SIZE = 3;
   basic_trainer.set_sample_size(SAMPLE_SIZE);
   ASSERT_EQ(basic_trainer.sample_size(), SAMPLE_SIZE);

   // Check default sample size to draw for evaluation
   size_t DEFAULT_SAMPLING_COUNT = TypeParam::DEFAULT_SAMPLING_COUNT;
   ASSERT_EQ(basic_trainer.sampling_count(), DEFAULT_SAMPLING_COUNT);

   // Set and test sample size
   size_t SAMPLING_COUNT = 2;
   basic_trainer.set_sampling_count(SAMPLING_COUNT);
   ASSERT_EQ(basic_trainer.sampling_count(), SAMPLING_COUNT);
}
 */
#endif //_NNEVALCONFIGTESTS_H_
