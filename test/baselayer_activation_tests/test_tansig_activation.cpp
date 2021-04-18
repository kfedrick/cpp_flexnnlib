//
// Created by kfedrick on 5/27/19.
//

#include <gtest/gtest.h>

#include "LayerActivationTestCase.h"

#include "TanSig.h"

using std::valarray;
using flexnnet::TanSig;

TEST_P(TestTanSigActivation, Activation)
{
   printf("TEST Case %s\n", GetParam());

   /*
    * Read a test case file. Each test case file contains a JSON basic_layer description
    * suitable for building and configuring a network basic_layer along with a set_weights of
    * test input/output pairs for that specific basic_layer configuration.
    */
   LayerActivationTestCase<TanSig> test_case;
   test_case.read(GetParam());

   flexnnet::LayerState lstate;
   std::valarray<double> errv;

   for (auto& item : test_case.samples)
   {
      errv.resize(test_case.layer_ptr->size());
      lstate.resize(test_case.layer_ptr->size(), item.input.size());

      test_case.layer_ptr->activate(item.input, lstate);
      test_case.layer_ptr->backprop(errv, lstate);

      const valarray<double>& layer_out = lstate.outputv;      //printResults(*test_case.layer_ptr, 9);

      // Check basic_layer output
      EXPECT_PRED3(vector_double_near, item.target.output, layer_out, 0.000000001) << "ruh roh";

      // Check dy_dnet
      EXPECT_PRED3(array_double_near, item.target.dAdN, lstate.dy_dnet, 0.000000001)
                  << prettyPrintArray("dy_dnet", lstate.dy_dnet);

      // Check dnet_dw
      EXPECT_PRED3(array_double_near, item.target.dNdW, lstate.dnet_dw, 0.000000001)
                  << prettyPrintArray("dnet_dw", lstate.dnet_dw);

      // Check dnet_dx
      EXPECT_PRED3(array_double_near, item.target.dNdI, lstate.dnet_dx, 0.000000001)
                  << prettyPrintArray("dnet_dx", lstate.dnet_dx);
   }
}

TEST_P(TestTanSigActivation, ActivateTwice)
{
   printf("TEST Case %s\n", GetParam());

   /*
    * Read a test case file. Each test case file contains a JSON basic_layer description
    * suitable for building and configuring a network basic_layer along with a set_weights of
    * test input/output pairs for that specific basic_layer configuration.
    */
   LayerActivationTestCase<TanSig> test_case;
   test_case.read(GetParam());

   flexnnet::LayerState lstate;
   std::valarray<double> errv;

   for (auto& item : test_case.samples)
   {
      errv.resize(test_case.layer_ptr->size());
      lstate.resize(test_case.layer_ptr->size(), item.input.size());

      test_case.layer_ptr->activate(item.input, lstate);
      test_case.layer_ptr->activate(item.input, lstate);
      test_case.layer_ptr->backprop(errv, lstate);

      const valarray<double>& layer_out = lstate.outputv;      //printResults(*test_case.layer_ptr, 9);

      // Check basic_layer output
      EXPECT_PRED3(vector_double_near, item.target.output, layer_out, 0.000000001) << "ruh roh";

      // Check dy_dnet
      EXPECT_PRED3(array_double_near, item.target.dAdN, lstate.dy_dnet, 0.000000001)
                  << prettyPrintArray("dy_dnet", lstate.dy_dnet);

      // Check dnet_dw
      EXPECT_PRED3(array_double_near, item.target.dNdW, lstate.dnet_dw, 0.000000001)
                  << prettyPrintArray("dnet_dw", lstate.dnet_dw);

      // Check dnet_dx
      EXPECT_PRED3(array_double_near, item.target.dNdI, lstate.dnet_dx, 0.000000001)
                  << prettyPrintArray("dnet_dx", lstate.dnet_dx);
   }
}