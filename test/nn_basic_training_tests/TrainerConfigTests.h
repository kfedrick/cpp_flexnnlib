//
// Created by kfedrick on 10/4/19.
//

#ifndef _TRAINERCONFIGTESTS_H_
#define _TRAINERCONFIGTESTS_H_

#include "gtest/gtest.h"

#include "BasicTrainerTests.h"

TYPED_TEST_P (BasicTrainerTests, TrainerConfig)
{
   std::cout << "\nBasicTrainerTests::TrainerConfig\n";

   TypeParam basic_trainer;

// ---   Set basic trainer configuration variables

// Check default max epochs
   size_t DEFAULT_MAX_EPOCHS = TypeParam::DEFAULT_MAX_EPOCHS;
   ASSERT_EQ(basic_trainer.max_epochs(), DEFAULT_MAX_EPOCHS);

// Set and check max epochs
   size_t MAX_EPOCHS = 2;
   basic_trainer.set_max_epochs(MAX_EPOCHS);
   ASSERT_EQ(basic_trainer.max_epochs(), MAX_EPOCHS);

// Test default batch mode
   size_t DEFAULT_BATCH_MODE = TypeParam::DEFAULT_BATCH_MODE;
   ASSERT_EQ(basic_trainer.batch_mode(), DEFAULT_BATCH_MODE);

// Set and test batch mode
   size_t BATCH_MODE = 13;
   basic_trainer.set_batch_mode(BATCH_MODE);
   ASSERT_EQ(basic_trainer.batch_mode(), BATCH_MODE);

// Test default error goal
   size_t DEFAULT_ERROR_GOAL = TypeParam::DEFAULT_ERROR_GOAL;
   ASSERT_EQ(basic_trainer.error_goal(), DEFAULT_ERROR_GOAL);

// Set and test error goal
   double ERR_GOAL = 1e-5;
   basic_trainer.set_error_goal(ERR_GOAL);
   ASSERT_EQ(basic_trainer.error_goal(), ERR_GOAL);

// Test default max validation failures
   size_t DEFAULT_MAX_VFAILURES = TypeParam::DEFAULT_MAX_VALIDATION_FAIL;
   ASSERT_EQ(basic_trainer.max_validation_failures(), DEFAULT_MAX_VFAILURES);

// Set and test max validation failures
   size_t MAX_VFAIL = 5;
   basic_trainer.set_max_validation_failures(MAX_VFAIL);
   ASSERT_EQ(basic_trainer.max_validation_failures(), MAX_VFAIL);

// Test default report frequency
   size_t DEFAULT_RFREQ = TypeParam::DEFAULT_REPORT_FREQ;
   ASSERT_EQ(basic_trainer.report_frequency(), DEFAULT_RFREQ);

// Set and test report frequency
   size_t RFREQ = 3;
   basic_trainer.set_report_frequency(RFREQ);
   ASSERT_EQ(basic_trainer.report_frequency(), RFREQ);

// Test default display frequency
   size_t DEFAULT_DFREQ = TypeParam::DEFAULT_DISPLAY_FREQ;
   ASSERT_EQ(basic_trainer.display_frequency(), DEFAULT_DFREQ);

// Set and test display frequency
   size_t DFREQ = 3;
   basic_trainer.set_display_frequency(DFREQ);
   ASSERT_EQ(basic_trainer.display_frequency(), DFREQ);

// Test default training runs
   size_t DEFAULT_RUNS = TypeParam::DEFAULT_TRAINING_RUNS;
   ASSERT_EQ(basic_trainer.training_runs(), DEFAULT_RUNS);

// Set and test training runs
   size_t RUNS = 3;
   basic_trainer.set_training_runs(RUNS);
   ASSERT_EQ(basic_trainer.training_runs(), RUNS);

   // Test default saved neural network limit
   size_t DEFAULT_SAVE_LIMIT = TypeParam::DEFAULT_SAVED_NNET_LIMIT;
   ASSERT_EQ(basic_trainer.saved_nnet_limit(), DEFAULT_SAVE_LIMIT);

// Set and test saved neural network limit
   size_t SAVE_LIMIT = 3;
   basic_trainer.set_saved_nnet_limit(SAVE_LIMIT);
   ASSERT_EQ(basic_trainer.saved_nnet_limit(), SAVE_LIMIT);
}

#endif //_TRAINERCONFIGTESTS_H_
