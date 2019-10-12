//
// Created by kfedrick on 6/19/19.
//

#ifndef _TEST_SINGLE_LAYER_NNET_H_
#define _TEST_SINGLE_LAYER_NNET_H_

#include <gtest/gtest.h>

#include <string>

#include "BasicLayer.h"
#include "PureLin.h"
#include "TanSig.h"
#include "RadBas.h"
#include "SoftMax.h"
#include "LogSig.h"

using std::string;
using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::PureLin;
using flexnnet::TanSig;
using flexnnet::RadBas;
using flexnnet::SoftMax;
using flexnnet::LogSig;

template<typename T>
class TestSingleLayerNNBuild : public ::testing::Test
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}
};

TYPED_TEST_CASE_P
(TestSingleLayerNNBuild);

typedef ::testing::Types<PureLin, TanSig, RadBas, SoftMax, LogSig> MyTypes;

#endif //_TEST_SINGLE_LAYER_NNET_H_
