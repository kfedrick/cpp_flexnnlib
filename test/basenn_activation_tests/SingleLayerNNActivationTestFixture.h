//
// Created by kfedrick on 2/22/21.
//

#ifndef _SINGLELAYERNNACTIVATIONTESTFIXTURE_H_
#define _SINGLELAYERNNACTIVATIONTESTFIXTURE_H_

#include <gtest/gtest.h>

#include <CommonTestFixtureFunctions.h>
#include "test/include/BaseNNActivationTestFixture.h"

#include "flexnnet.h"
#include "BasicLayer.h"
#include "PureLin.h"
#include "TanSig.h"
#include "RadBas.h"
#include "SoftMax.h"
#include "LogSig.h"

#include "BaseNeuralNet.h"

#define TESTCASE_PATH "test/basenn_activation_tests/samples/"



template<typename T>
class SingleLayerNNActivationTestFixture : public CommonTestFixtureFunctions, public BaseNNActivationTestFixture<T>, public ::testing::Test
{

public:

   struct ActivationTestCase
   {
      size_t layer_sz;
      size_t input_sz;
      flexnnet::Array2D<double> weights;

      flexnnet::ValarrMap input;
      flexnnet::ValarrMap target_output;
   };

   struct DerivativesTestCase
   {
      size_t layer_sz;
      size_t input_sz;
      flexnnet::Array2D<double> weights;

      flexnnet::ValarrMap input;

      struct Target
      {
         flexnnet::Array2D<double> dAdN;
         flexnnet::Array2D<double> dNdW;
         flexnnet::Array2D<double> dNdI;
      };

      Target target;
   };

public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   std::vector<ActivationTestCase> read_activation_samples(std::string _fpath);
   std::vector<DerivativesTestCase> read_derivatives_samples(std::string _fpath);

   void create_nnet(const ActivationTestCase& _testcase);
   std::shared_ptr<flexnnet::BaseNeuralNet> create_deriv_nnet(const DerivativesTestCase& _testcase);

   std::shared_ptr<flexnnet::BaseNeuralNet> nnet;
};
TYPED_TEST_CASE_P (SingleLayerNNActivationTestFixture);

typedef ::testing::Types<flexnnet::PureLin,
                         flexnnet::TanSig,
                         flexnnet::LogSig,
                         flexnnet::SoftMax,
                         flexnnet::RadBas> MyTypes;

#endif //_SINGLELAYERNNACTIVATIONTESTFIXTURE_H_
