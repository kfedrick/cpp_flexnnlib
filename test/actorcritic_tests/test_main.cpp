//
// Created by kfedrick on 6/25/19.
//

#include "gtest/gtest.h"

//#include "BasicACTests.h"
#include "ACActivationTests.h"
//#include "ACTrainerTest.h"

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);

   srand (time(NULL));
   return RUN_ALL_TESTS();
}