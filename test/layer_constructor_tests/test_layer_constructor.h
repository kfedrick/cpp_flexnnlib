//
// Created by kfedrick on 6/17/19.
//

#ifndef _TEST_LAYER_CONSTRUCTOR_H_
#define _TEST_LAYER_CONSTRUCTOR_H_

#include <gtest/gtest.h>

#include "Array2D.h"
#include "BasicLayer.h"

class TestLayerConstructors : public ::testing::TestWithParam<const char*>
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}
};

#endif //_TEST_LAYER_CONSTRUCTOR_H_
