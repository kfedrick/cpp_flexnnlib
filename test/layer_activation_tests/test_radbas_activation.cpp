//
// Created by kfedrick on 5/27/19.
//
#include <gtest/gtest.h>
#include <memory>

#include "test_layer_activation.h"
#include "LayerActivationTestCase.h"

#include "RadBas.h"

using std::valarray;
using flexnnet::RadBas;

TEST_P(TestRadBasActivation, Activation)
{
   printf ("TEST Case %s\n", GetParam ());

   /*
    * Read a test case file. Each test case file contains a JSON layer description
    * suitable for building and configuring a network layer along with a set of
    * test input/output pairs for that specific layer configuration.
    */
   LayerActivationTestCase<RadBas> test_case;
   test_case.read (GetParam ());


   for (auto &item : test_case.samples)
   {
      const valarray<double> &layer_out = test_case.layer_ptr->activate (item.input);
      printResults (*test_case.layer_ptr, 9);

      // Check layer output
      EXPECT_PRED3(vector_double_near, item.target.output, layer_out, 0.000000001) << "ruh roh";

      // Check dAdN
      EXPECT_PRED3(array_double_near, item.target.dAdN, test_case.layer_ptr->get_dAdN(), 0.000000001) << "ruh roh";

      // Check dNdW
      EXPECT_PRED3(array_double_near, item.target.dNdW, test_case.layer_ptr->get_dNdW(), 0.000000001) << printArray("dNdW", test_case.layer_ptr->get_dNdW());

      // Check dNdI
      EXPECT_PRED3(array_double_near, item.target.dNdI, test_case.layer_ptr->get_dNdI(), 0.000000001) << printArray("dNdI", test_case.layer_ptr->get_dNdI());
   }
}
