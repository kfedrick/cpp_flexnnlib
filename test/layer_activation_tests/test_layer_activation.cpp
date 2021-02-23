//
// Created by kfedrick on 5/4/19.
//
// Test basiclayer activation by:
//
// 1. Verifying expected outputs for a set_weights of fixed inputs and basiclayer weights.
//
// 2. Making small alterations to inputs, weights and biases and
//    verifying that it results in a suitably small change in the basiclayer output.
//

#include "gtest/gtest.h"

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);

   return RUN_ALL_TESTS();
}

