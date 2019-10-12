//
// Created by kfedrick on 6/19/19.
//

#ifndef _TEST_NN_SERIALIZATION_H_
#define _TEST_NN_SERIALIZATION_H_

#include <gtest/gtest.h>

#include "Array2D.h"
#include "BasicLayer.h"
#include "BasicNeuralNet.h"
#include "BasicNeuralNetFactory.h"
#include "BasicNeuralNetSerializer.h"
#include "NeuralNetSerializer.h"
#include "TestLayer.h"

class TestBasicNeuralNet : public TestLayer, public ::testing::TestWithParam<const char*>
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}
};

#endif //_TEST_NN_SERIALIZATION_H_
