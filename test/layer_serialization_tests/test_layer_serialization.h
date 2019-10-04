//
// Created by kfedrick on 6/17/19.
//


#ifndef _TEST_LAYER_SERIALIZATION_H_
#define _TEST_LAYER_SERIALIZATION_H_

#include <gtest/gtest.h>
#include "TestLayer.h"

#include "Array2D.h"
#include "BasicLayer.h"

class TestLayerSerialization : public TestLayer, public ::testing::TestWithParam<const char *>
{
public:
   virtual void SetUp ()
   {}
   virtual void TearDown ()
   {}
};

#endif //_TEST_LAYER_SERIALIZATION_H_
