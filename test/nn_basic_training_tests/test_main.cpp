//
// Created by kfedrick on 6/25/19.
//

#include "gtest/gtest.h"

#include "SupervisedTrainerTestFixture.h"
#include "TrainerConfigTests.h"
#include "ClassifierTrainingTests.h"

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);

   return RUN_ALL_TESTS();
}