//
// Created by kfedrick on 5/21/19.
//

#ifndef _TEST_LAYER_DERIVATIVES_H_
#define _TEST_LAYER_DERIVATIVES_H_

#include <gtest/gtest.h>
#include <map>
#include <vector>
#include <cmath>
#include <string>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>
#include <layers/include/RadialBasisTrans.h>

#include "Layer.h"
#include "PureLinTrans.h"
#include "LogSigTrans.h"

#include "JSONEncoder.h"
#include "PureLinSerializer.h"
#include "LogSigSerializer.h"
#include "TanSigSerializer.h"
#include "RadialBasisSerializer.h"

#include "LayerDerivTestCase.h"
#include "PureLinDerivTestCase.h"

class TestLayerDerivatives : public ::testing::TestWithParam<const char *>
{
public:

   virtual void SetUp ()
   {
   }

   virtual void TearDown ()
   {
   }

   static bool vector_double_near (const std::vector<double> &_target, const std::vector<double> &_test, double _epsilon);

};

bool TestLayerDerivatives::vector_double_near (const std::vector<double> &_target, const std::vector<double> &_test, double _epsilon)
{
   if (_target.size () != _test.size ())
      return false;

   for (unsigned int i = 0; i < _target.size (); i++)
      if (fabs (_target[i] - _test[i]) > _epsilon)
         return false;

   return true;
}



TEST_F(TestLayerDerivatives, dAdN)
{
   const int OUT_SZ = 2;
   const int IN_SZ = 3;
   const std::string NAME = "test";

   flexnnet::Layer<flexnnet::PureLinTrans> layer (OUT_SZ, NAME);

   ASSERT_EQ(OUT_SZ, layer.size ());
   ASSERT_EQ(0, layer.input_size ());
   ASSERT_EQ(NAME, layer.name ());

   layer.resize_input_vector (IN_SZ);
   ASSERT_EQ(IN_SZ, layer.input_size ());

   layer.set_gain (1.6);

   // weight vector dimension (layer output size, raw input size +1)
   // 1 extra column to account for biases
   //
   flexnnet::Array<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw[0][0] = 1.0;
   initw[1][2] = 1.0;
   layer.layer_weights.set_weights (initw);

   printf ("weights\n");
   for (unsigned int i = 0; i < layer.layer_weights.const_weights_ref.rowDim (); i++)
   {
      for (unsigned int j = 0; j < layer.layer_weights.const_weights_ref.colDim (); j++)
         printf ("%7.5f ", layer.layer_weights.const_weights_ref[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   std::vector<double> in = {1.0, 0.0, 0.33};
   const std::vector<double> &layer_out = layer.activate (in);

   printf ("PureLinLayer output\n");
   for (unsigned int i = 0; i < layer_out.size (); i++)
      printf ("%7.5f ", layer_out[i]);
   printf ("\n\n");


   printf ("dAdN\n");
   const flexnnet::Array<double>& dAdN = layer.get_dAdN();
   for (unsigned int i = 0; i < dAdN.rowDim (); i++)
   {
      for (unsigned int j = 0; j < dAdN.colDim (); j++)
         printf ("%7.5f ", dAdN[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");


   printf ("dNdW\n");
   const flexnnet::Array<double>& dNdW = layer.get_dNdW();
   for (unsigned int i = 0; i < dNdW.rowDim (); i++)
   {
      for (unsigned int j = 0; j < dNdW.colDim (); j++)
         printf ("%7.5f ", dNdW[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   printf ("dNdI\n");
   const flexnnet::Array<double>& dNdI = layer.get_dNdI();
   for (unsigned int i = 0; i < dNdI.rowDim (); i++)
   {
      for (unsigned int j = 0; j < dNdI.colDim (); j++)
         printf ("%7.5f ", dNdI[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");
}

TEST_F(TestLayerDerivatives, RBF)
{
   const int OUT_SZ = 2;
   const int IN_SZ = 3;
   const std::string NAME = "test";

   flexnnet::Layer<flexnnet::RadialBasisTrans> layer (OUT_SZ, NAME);

   ASSERT_EQ(OUT_SZ, layer.size ());
   ASSERT_EQ(0, layer.input_size ());
   ASSERT_EQ(NAME, layer.name ());

   layer.resize_input_vector (IN_SZ);
   ASSERT_EQ(IN_SZ, layer.input_size ());

   // weight vector dimension (layer output size, raw input size +1)
   // 1 extra column to account for biases
   //
   flexnnet::Array<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw[0][0] = 1.0;
   initw[1][2] = 1.0;
   layer.layer_weights.set_weights (initw);

   printf ("weights\n");
   for (unsigned int i = 0; i < layer.layer_weights.const_weights_ref.rowDim (); i++)
   {
      for (unsigned int j = 0; j < layer.layer_weights.const_weights_ref.colDim (); j++)
         printf ("%7.5f ", layer.layer_weights.const_weights_ref[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   std::vector<double> in = {1.0, 0.0, 0.33};
   const std::vector<double> &layer_out = layer.activate (in);

   printf ("PureLinLayer output\n");
   for (unsigned int i = 0; i < layer_out.size (); i++)
      printf ("%7.5f ", layer_out[i]);
   printf ("\n\n");


   printf ("dAdN\n");
   const flexnnet::Array<double>& dAdN = layer.get_dAdN();
   for (unsigned int i = 0; i < dAdN.rowDim (); i++)
   {
      for (unsigned int j = 0; j < dAdN.colDim (); j++)
         printf ("%7.5f ", dAdN[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");


   printf ("dNdW\n");
   const flexnnet::Array<double>& dNdW = layer.get_dNdW();
   for (unsigned int i = 0; i < dNdW.rowDim (); i++)
   {
      for (unsigned int j = 0; j < dNdW.colDim (); j++)
         printf ("%7.5f ", dNdW[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   printf ("dNdI\n");
   const flexnnet::Array<double>& dNdI = layer.get_dNdI();
   for (unsigned int i = 0; i < dNdI.rowDim (); i++)
   {
      for (unsigned int j = 0; j < dNdI.colDim (); j++)
         printf ("%7.5f ", dNdI[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");
}

TEST_F(TestLayerDerivatives, dAdN2)
{
   /*
    * Read a test case file. Each test case file contains a JSON layer description
    * suitable for building and configuring a network layer along with a set of
    * test input/output pairs for that specific layer configuration.
    */
   flexnnet::PureLinDerivTestCase test_case;
   test_case.read ("purelin_derivatives_test1.json");
   //test_case.read (GetParam ());


   /*
    * Just for fun, print the input/output samples.
    */
   std::vector<flexnnet::LayerDerivTestCase::LayerDerivTestSample> test_pairs = test_case.samples;

   for (auto &item : test_case.samples)
   {
      printf ("\n");
      for (auto &val : item.input)
         printf ("%f ", val);

      printf ("\n\n");
      printf ("target dAdN 1\n");
      for (unsigned int i = 0; i < item.target_dAdN.rowDim (); i++)
      {
         for (unsigned int j = 0; j < item.target_dAdN.colDim (); j++)
            printf ("%7.5f ", item.target_dAdN[i][j]);
         printf ("\n");
      }
      printf ("---\n\n");

      printf ("\n\n");
      printf ("target dNdW 1\n");
      for (unsigned int i = 0; i < item.target_dNdW.rowDim (); i++)
      {
         for (unsigned int j = 0; j < item.target_dNdW.colDim (); j++)
            printf ("%7.5f ", item.target_dNdW[i][j]);
         printf ("\n");
      }
      printf ("---\n\n");

      printf ("\n\n");
      printf ("target dNdI 1\n");
      for (unsigned int i = 0; i < item.target_dNdI.rowDim (); i++)
      {
         for (unsigned int j = 0; j < item.target_dNdI.colDim (); j++)
            printf ("%7.5f ", item.target_dNdI[i][j]);
         printf ("\n");
      }
      printf ("---\n\n");
   }
   printf ("\n\n");
}


#endif //_TEST_LAYER_DERIVATIVES_H_
