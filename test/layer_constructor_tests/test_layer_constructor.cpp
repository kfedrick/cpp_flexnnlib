//
// Created by kfedrick on 6/17/19.
//

#include "gtest/gtest.h"

#include "test_purelin_constructor.h"
#include "test_tansig_constructor.h"
#include "test_logsig_constructor.h"
#include "test_radbas_constructor.h"
#include "test_softmax_constructor.h"
#include "Evaluator.h"
#include "RMSError.h"

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);

   return RUN_ALL_TESTS();
}

