//
// Created by kfedrick on 2/27/21.
//

#ifndef _FANINNNACTIVATIONTESTFIXTURE_H_
#define _FANINNNACTIVATIONTESTFIXTURE_H_

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
class FaninNNActivationTestFixture : public CommonTestFixtureFunctions, public BaseNNActivationTestFixture<T>, public ::testing::Test
{
public:
   struct TestCase
   {
      size_t hlayer1_sz;
      size_t hlayer1_input_sz;

      size_t hlayer2_sz;
      size_t hlayer2_input_sz;

      flexnnet::Array2D<double> hlayer1_weights;
      flexnnet::Array2D<double> hlayer2_weights;
      flexnnet::Array2D<double> output_weights;

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
TYPED_TEST_CASE_P (FaninNNActivationTestFixture);

typedef ::testing::Types<flexnnet::PureLin,
flexnnet::TanSig,
flexnnet::LogSig,
flexnnet::SoftMax,
flexnnet::RadBas> MyTypes;

#endif //_FANINNNACTIVATIONTESTFIXTURE_H_
