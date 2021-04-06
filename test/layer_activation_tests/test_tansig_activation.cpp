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

   for (auto& item : test_case.samples)
   {
      const valarray<double>& layer_out = test_case.layer_ptr->activate(item.input);
      //printResults(*test_case.layer_ptr, 9);

      // Check basic_layer output
      EXPECT_PRED3(vector_double_near, item.target.output, layer_out, 0.000000001) << "ruh roh";

      // Check dy_dnet
      EXPECT_PRED3(array_double_near, item.target.dAdN, test_case.layer_ptr->get_dy_dnet(), 0.000000001)
                  << prettyPrintArray("dy_dnet", test_case.layer_ptr->get_dy_dnet());

      // Check dnet_dw
      EXPECT_PRED3(array_double_near, item.target.dNdW, test_case.layer_ptr->get_dnet_dw(), 0.000000001)
                  << prettyPrintArray("dnet_dw", test_case.layer_ptr->get_dnet_dw());

      // Check dnet_dx
      EXPECT_PRED3(array_double_near, item.target.dNdI, test_case.layer_ptr->get_dnet_dx(), 0.000000001)
                  << prettyPrintArray("dnet_dx", test_case.layer_ptr->get_dnet_dx());
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

   for (auto& item : test_case.samples)
   {
      test_case.layer_ptr->activate(item.input);
      const valarray<double>& layer_out = test_case.layer_ptr->activate(item.input);
      //printResults(*test_case.layer_ptr, 9);

      // Check basic_layer output
      EXPECT_PRED3(vector_double_near, item.target.output, layer_out, 0.000000001) << "ruh roh";

      // Check dy_dnet
      EXPECT_PRED3(array_double_near, item.target.dAdN, test_case.layer_ptr->get_dy_dnet(), 0.000000001)
                  << prettyPrintArray("dy_dnet", test_case.layer_ptr->get_dy_dnet());

      // Check dnet_dw
      EXPECT_PRED3(array_double_near, item.target.dNdW, test_case.layer_ptr->get_dnet_dw(), 0.000000001)
                  << prettyPrintArray("dnet_dw", test_case.layer_ptr->get_dnet_dw());

      // Check dnet_dx
      EXPECT_PRED3(array_double_near, item.target.dNdI, test_case.layer_ptr->get_dnet_dx(), 0.000000001)
                  << prettyPrintArray("dnet_dx", test_case.layer_ptr->get_dnet_dx());
   }
}