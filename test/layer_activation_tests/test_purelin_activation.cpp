//
// Created by kfedrick on 5/4/19.
//

#include <gtest/gtest.h>

#include "LayerActivationTestCase.h"

#include "PureLin.h"

using std::cout;
using std::string;
using std::valarray;
using std::unique_ptr;

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::PureLin;

TEST_P(TestPureLinActivation, Activation)
{
   printf("TEST Case %s\n", GetParam());

   /*
    * Read a test case file. Each test case file contains a JSON layer description
    * suitable for building and configuring a network layer along with a set_weights of
    * test input/output pairs for that specific layer configuration.
    */
   LayerActivationTestCase<flexnnet::PureLin> test_case;
   test_case.read(GetParam());

   for (auto& item : test_case.samples)
   {
      const valarray<double>& layer_out = test_case.layer_ptr->activate(item.input);
      printResults(*test_case.layer_ptr);

      // Check layer output
      EXPECT_PRED3(vector_double_near, item.target.output, layer_out, 0.000000001) << "ruh roh";

      // Check dAdN
      EXPECT_PRED3(array_double_near, item.target.dAdN, test_case.layer_ptr->get_dAdN(), 0.000000001) << "ruh roh";

      // Check dNdW
      EXPECT_PRED3(array_double_near, item.target.dNdW, test_case.layer_ptr->get_dNdW(), 0.000000001)
                  << prettyPrintArray("dNdW", test_case.layer_ptr->get_dNdW());

      // Check dNdI
      EXPECT_PRED3(array_double_near, item.target.dNdI, test_case.layer_ptr->get_dNdI(), 0.000000001)
                  << prettyPrintArray("dNdI", test_case.layer_ptr->get_dNdI());
   }
}
