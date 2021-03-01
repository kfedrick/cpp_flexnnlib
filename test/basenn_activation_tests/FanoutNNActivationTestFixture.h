//
// Created by kfedrick on 2/28/21.
//

#ifndef _FANOUTNNACTIVATIONTESTFIXTURE_H_
#define _FANOUTNNACTIVATIONTESTFIXTURE_H_
#include <gtest/gtest.h>

#include "CommonTestFixtureFunctions.h"
#include "BaseNNActivationTestFixture.h"

#include "flexnnet.h"
#include "PureLin.h"
#include "TanSig.h"
#include "RadBas.h"
#include "SoftMax.h"
#include "LogSig.h"

#include "BaseNeuralNet.h"

template<typename T>
class FanoutNNActivationTestFixture : public CommonTestFixtureFunctions, public BaseNNActivationTestFixture<T>, public ::testing::Test
{
public:
   struct TestCase
   {
      size_t input1_sz;
      size_t input2_sz;

      size_t hlayer_sz;
      flexnnet::Array2D<double> hlayer_weights;

      size_t olayer1_sz;
      size_t olayer2_sz;

      flexnnet::Array2D<double> olayer1_weights;
      flexnnet::Array2D<double> olayer2_weights;

      flexnnet::NNetIO_Typ input;
      flexnnet::NNetIO_Typ target_output;
   };

public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   std::vector<TestCase> read_samples(std::string _fpath);
   void create_nnet(const TestCase& _testcase);

   std::shared_ptr<flexnnet::BaseNeuralNet> nnet;
};
TYPED_TEST_CASE_P (FanoutNNActivationTestFixture);

typedef ::testing::Types<flexnnet::PureLin,
                         flexnnet::TanSig,
                         flexnnet::LogSig,
                         flexnnet::SoftMax,
                         flexnnet::RadBas> MyTypes;

#endif //_FANOUTNNACTIVATIONTESTFIXTURE_H_
