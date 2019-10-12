//
// Created by kfedrick on 6/25/19.
//

#include "gtest/gtest.h"

#include "BasicTrainerTests.h"

#include "TrainerConfigTests.h"
#include "NNEvalConfigTests.h"
#include "BasicTrainerSeqTests.h"

//REGISTER_TYPED_TEST_CASE_P(BasicTrainerTests, TrainerConfig, NNEvalConfig, BasicSequence  );
REGISTER_TYPED_TEST_CASE_P
(BasicTrainerTests, TrainerConfig, BasicSequence);
INSTANTIATE_TYPED_TEST_CASE_P
(My, BasicTrainerTests, MyTypes);

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);

   return RUN_ALL_TESTS();
}