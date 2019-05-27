//
// Created by kfedrick on 5/4/19.
//

#ifndef TEST_PURELIN_ACTIVATION_H
#define TEST_PURELIN_ACTIVATION_H

#include <gtest/gtest.h>
#include <map>
#include <vector>
#include <cmath>

#include "flexneuralnet.h"
#include "NetSum.h"
#include "Layer.h"
#include "Array.h"
#include "BasicLayer.h"
#include "SoftMax.h"

#include "Base.h"
#include "Derived.h"

#include "LayerWeights.h"
#include "LayerInfo.h"

#include "PureLinTrans.h"
#include "LogSigTrans.h"
#include "TanSigTrans.h"
#include "RadialBasisTrans.h"

#include "JSONEncoder.h"

#include "LayerActivationTestCase.h"
#include "PureLinActivationTestCase.h"

#include "PureLinSerializer.h"
#include "LogSigSerializer.h"
#include "TanSigSerializer.h"
#include "RadialBasisSerializer.h"


#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>

using namespace std;
using namespace flexnnet;
using namespace rapidjson;

//class TestPureLinLayer : public ::testing::Test
class TestPureLinLayer : public ::testing::TestWithParam<const char *>
{
public:

   virtual void SetUp ()
   {
   }

   virtual void TearDown ()
   {
   }

protected:

   Layer<PureLinTrans> *createPureLin (const PureLinActivationTestCase &_testcase);
   void assert_vector_double_near (const vector<double> &_target, const vector<double> &_test, double _epsilon);
   static bool vector_double_near (const vector<double> &_target, const vector<double> &_test, double _epsilon);
};

INSTANTIATE_TEST_CASE_P
(InstantiationName, TestPureLinLayer, ::testing::Values ("purelin_activity_test1.json", "purelin_activity_test2.json"));

void
TestPureLinLayer::assert_vector_double_near (const vector<double> &_target, const vector<double> &_test, double _epsilon)
{
   ASSERT_EQ(_target.size (), _test.size ());
   for (unsigned int i = 0; i < _target.size (); i++)
      ASSERT_NEAR(_target[i], _test[i], _epsilon);
}

bool TestPureLinLayer::vector_double_near (const vector<double> &_target, const vector<double> &_test, double _epsilon)
{
   if (_target.size () != _test.size ())
      return false;

   for (unsigned int i = 0; i < _target.size (); i++)
      if (fabs (_target[i] - _test[i]) > _epsilon)
         return false;

   return true;
}

Layer<PureLinTrans> *TestPureLinLayer::createPureLin (const PureLinActivationTestCase &_testcase)
{
   /*
    * Assume we are handed a description of a Layer<PureLinTrans> for now and create
    * the layer with the specified size and input size.
    */
   Layer<PureLinTrans> *layer_ptr = new Layer<PureLinTrans> (_testcase.layer_size, _testcase.layer_name);
   layer_ptr->resize_input_vector (_testcase.input_size);

   // Set the layer weights according to the descriptions
   layer_ptr->layer_weights.set_weights (_testcase.weights);

   // Set the transfer function specific parameters
   layer_ptr->set_gain (_testcase.gain);

   return layer_ptr;
}

TEST_P(TestPureLinLayer, LayerJSONRead)
{
   printf ("TEST Case %s\n", GetParam ());
   /*
    * Read a test case file. Each test case file contains a JSON layer description
    * suitable for building and configuring a network layer along with a set of
    * test input/output pairs for that specific layer configuration.
    */
   PureLinActivationTestCase test_case;
   //test_case.read ("purelin_activity_test1.json");
   test_case.read (GetParam ());


   /*
    * Just for fun, print the input/output samples.
    */
   vector<LayerActivationTestCase::LayerActivationTestPair> test_pairs = test_case.samples;

   for (auto &item : test_case.samples)
   {
      printf ("\n");
      for (auto &val : item.input)
         printf ("%f ", val);
      printf (" => ");

      for (auto &val : item.target)
         printf ("%f ", val);
   }
   printf ("\n\n");

   // Create and configure a layer from the specified configuration.
   Layer<PureLinTrans> *layer_ptr = createPureLin (test_case);

   printf ("\n\n");
   printf ("weights 1\n");
   for (unsigned int i = 0; i < layer_ptr->layer_weights.const_weights_ref.rowDim (); i++)
   {
      for (unsigned int j = 0; j < layer_ptr->layer_weights.const_weights_ref.colDim (); j++)
         printf ("%7.5f ", layer_ptr->layer_weights.const_weights_ref[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   printf ("Do something else\n");
   for (auto &item : test_case.samples)
   {
      const vector<double> &layer_out = layer_ptr->activate (item.input);
      EXPECT_PRED3(vector_double_near, item.target, layer_out, 0.000000001) << "ruh roh";
   }

}


TEST_F(TestPureLinLayer, FunctionalTest1)
{
   Base b;
   b.doit ();
}

TEST_F(TestPureLinLayer, FunctionalTest2)
{
   Derived d;
   Base b = d;
   b.calcit ();
}

TEST_F(TestPureLinLayer, LogSig)
{
   const int OUT_SZ = 2;
   const int IN_SZ = 3;
   const string NAME = "test";

   Layer<LogSigTrans> layer (OUT_SZ, NAME);

   printf ("\n\n");
   printf ("weights 1\n");
   for (unsigned int i = 0; i < layer.layer_weights.const_weights_ref.rowDim (); i++)
   {
      for (unsigned int j = 0; j < layer.layer_weights.const_weights_ref.colDim (); j++)
         printf ("%7.5f ", layer.layer_weights.const_weights_ref[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   ASSERT_EQ(OUT_SZ, layer.size ());
   ASSERT_EQ(0, layer.input_size ());
   ASSERT_EQ(NAME, layer.name ());

   layer.resize_input_vector (IN_SZ);
   ASSERT_EQ(IN_SZ, layer.input_size ());

   layer.set_gain (1.6);

   printf ("weights 2\n");
   for (unsigned int i = 0; i < layer.layer_weights.const_weights_ref.rowDim (); i++)
   {
      for (unsigned int j = 0; j < layer.layer_weights.const_weights_ref.colDim (); j++)
         printf ("%7.5f ", layer.layer_weights.const_weights_ref[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   // weight vector dimension (layer output size, raw input size +1)
   // 1 extra column to account for biases
   //
   Array<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw[0][0] = 1.0;
   initw[1][2] = 1.0;
   layer.layer_weights.set_weights (initw);

   printf ("weights 3\n");
   for (unsigned int i = 0; i < layer.layer_weights.const_weights_ref.rowDim (); i++)
   {
      for (unsigned int j = 0; j < layer.layer_weights.const_weights_ref.colDim (); j++)
         printf ("%7.5f ", layer.layer_weights.const_weights_ref[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   vector<double> in = {1.0, 0.0, 0.33};
   const vector<double> &layer_out = layer.activate (in);

   printf ("PureLinLayer output\n");
   for (unsigned int i = 0; i < layer_out.size (); i++)
      printf ("%7.5f ", layer_out[i]);
   printf ("\n\n");

   BasicLayer blayer = layer;

   const vector<double> &blayer_out = layer.activate (in);

   printf ("BasicLayer output\n");
   for (unsigned int i = 0; i < blayer_out.size (); i++)
      printf ("%7.5f ", blayer_out[i]);
   printf ("\n\n");

   std::string jsonstr = layer.toJSONString ();
   std::cout << "\n\n" << jsonstr << std::endl;

   std::string basejson = blayer.toJSONString ();
   std::cout << "\n\n" << basejson << std::endl;
}

TEST_F(TestPureLinLayer, TestWriteWeights)
{
   const int OUT_SZ = 2;
   const int IN_SZ = 3;
   const string NAME = "test";

   Layer<LogSigTrans> layer (OUT_SZ, NAME);
   layer.resize_input_vector (IN_SZ);

   // weight vector dimension (layer output size, raw input size +1)
   // 1 extra column to account for biases
   //
   Array<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw[0][0] = 1.0;
   initw[1][2] = 1.0;
   layer.layer_weights.set_weights (initw);

   vector<double> in = {1.0, 0.0, 0.33};
   const vector<double> &layer_out = layer.activate (in);

}

TEST_F(TestPureLinLayer, TestPureLin2)
{
   const int OUT_SZ = 2;
   const int IN_SZ = 3;
   const string NAME = "test";

   Layer<PureLinTrans> layer (OUT_SZ, NAME);
   layer.resize_input_vector (IN_SZ);

   // weight vector dimension (layer output size, raw input size +1)
   // 1 extra column to account for biases
   //
   Array<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw[0][0] = 1.0;
   initw[1][2] = 1.0;
   layer.layer_weights.set_weights (initw);

   layer.set_gain (0.33);

   std::string jsonstr = layer.toJSONString ();
   std::cout << "\n\n" << jsonstr << std::endl;

   BasicLayer &blayer = layer;
   std::string basejson = blayer.toJSONString ();
   std::cout << "\n\n" << basejson << std::endl;

   vector<double> in = {1.0, 0.0, 0.33};
   const vector<double> &layer_out = layer.activate (in);

   printf ("netout\n");
   for (unsigned int i = 0; i < layer_out.size (); i++)
      printf ("%7.5f ", layer_out[i]);
   printf ("\n\n");
}


TEST_F(TestPureLinLayer, RadialBasis)
{
   const int OUT_SZ = 2;
   const int IN_SZ = 3;
   const string NAME = "test";

   Layer<RadialBasisTrans> layer (OUT_SZ, NAME);
   layer.resize_input_vector (IN_SZ);

   // weight vector dimension (layer output size, raw input size +1)
   // 1 extra column to account for biases
   //
   Array<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw[0][0] = 1.0;
   initw[1][2] = 1.0;
   layer.layer_weights.set_weights (initw);

   std::string jsonstr = layer.toJSONString ();
   std::cout << "\n\n" << jsonstr << std::endl;

   BasicLayer &blayer = layer;
   std::string basejson = blayer.toJSONString ();
   std::cout << "\n\n" << basejson << std::endl;

   vector<double> in = {1.0, 0.0, 0.33};
   const vector<double> &layer_out = layer.activate (in);

   printf ("netout\n");
   for (unsigned int i = 0; i < layer_out.size (); i++)
      printf ("%7.5f ", layer_out[i]);
   printf ("\n\n");
}



TEST_F(TestPureLinLayer, TanSig)
{
   const int OUT_SZ = 2;
   const int IN_SZ = 3;
   const string NAME = "test";

   Layer<TanSigTrans> layer (OUT_SZ, NAME);
   layer.resize_input_vector (IN_SZ);

   layer.set_gain (0.666);

   // weight vector dimension (layer output size, raw input size +1)
   // 1 extra column to account for biases
   //
   Array<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw[0][0] = 1.0;
   initw[1][2] = 1.0;
   layer.layer_weights.set_weights (initw);

   std::string jsonstr = layer.toJSONString ();
   std::cout << "\n\n" << jsonstr << std::endl;

   BasicLayer &blayer = layer;
   std::string basejson = blayer.toJSONString ();
   std::cout << "\n\n" << basejson << std::endl;

   vector<double> in = {1.0, 0.0, 0.33};
   const vector<double> &layer_out = layer.activate (in);

   printf ("netout\n");
   for (unsigned int i = 0; i < layer_out.size (); i++)
      printf ("%7.5f ", layer_out[i]);
   printf ("\n\n");
}

TEST_F(TestPureLinLayer, LogSig3)
{
   const int OUT_SZ = 3;
   const int IN_SZ = 3;
   const string NAME = "test";

   Layer<LogSigTrans> layer (OUT_SZ, NAME);
   layer.resize_input_vector (IN_SZ);

   layer.set_gain (0.666);

   // weight vector dimension (layer output size, raw input size +1)
   // 1 extra column to account for biases
   //
   Array<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw[0][0] = 1.0;
   initw[1][2] = 1.0;
   layer.layer_weights.set_weights (initw);

   std::string jsonstr = layer.toJSONString ();
   std::cout << "\n\n" << jsonstr << std::endl;

   BasicLayer &blayer = layer;
   std::string basejson = blayer.toJSONString ();
   std::cout << "\n\n" << basejson << std::endl;

   vector<double> in = {1.0, 0.0, 0.33};
   const vector<double> &layer_out = layer.activate (in);

   printf ("netout\n");
   for (unsigned int i = 0; i < layer_out.size (); i++)
      printf ("%7.5f ", layer_out[i]);
   printf ("\n\n");
}

TEST_F(TestPureLinLayer, PureLinDeserialize)
{
   const std::string NAME = "test";
   const unsigned int OUT_SZ = 2;
   const unsigned int RAWIN_SZ = 3;
   const double GAIN = 0.33;

   std::string json = "{\"name\":\"test\",\"topology\":{\"layer_size\":2,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::PureLinTrans\",\"parameters\":{\"gain\":0.33}}}";
   const Layer<PureLinTrans>& layer = LayerSerializer< Layer<PureLinTrans> >::parse(json);

   ASSERT_EQ(OUT_SZ, layer.size ());
   ASSERT_EQ(RAWIN_SZ, layer.input_size ());
   ASSERT_EQ(NAME, layer.name ());
   ASSERT_EQ(GAIN, layer.get_gain());

   for (unsigned int i = 0; i < layer.layer_weights.const_weights_ref.rowDim (); i++)
   {
      for (unsigned int j = 0; j < layer.layer_weights.const_weights_ref.colDim (); j++)
         printf ("%7.5f ", layer.layer_weights.const_weights_ref[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   std::string new_json = LayerSerializer< Layer<PureLinTrans> >::toJSON(layer);
   std::cout << "\n\n" << new_json << std::endl;
}

TEST_F(TestPureLinLayer, TanSigDeserialize)
{
   const std::string NAME = "test";
   const unsigned int OUT_SZ = 2;
   const unsigned int RAWIN_SZ = 3;
   const double GAIN = 0.33;

   std::string json = "{\"name\":\"test\",\"topology\":{\"layer_size\":2,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::TanSigTrans\",\"parameters\":{\"gain\":0.33}}}";
   const Layer<TanSigTrans>& layer = LayerSerializer< Layer<TanSigTrans> >::parse(json);

   ASSERT_EQ(OUT_SZ, layer.size ());
   ASSERT_EQ(RAWIN_SZ, layer.input_size ());
   ASSERT_EQ(NAME, layer.name ());
   ASSERT_EQ(GAIN, layer.get_gain());

   for (unsigned int i = 0; i < layer.layer_weights.const_weights_ref.rowDim (); i++)
   {
      for (unsigned int j = 0; j < layer.layer_weights.const_weights_ref.colDim (); j++)
         printf ("%7.5f ", layer.layer_weights.const_weights_ref[i][j]);
      printf ("\n");
   }
   printf ("---\n\n");

   std::string new_json = LayerSerializer< Layer<TanSigTrans> >::toJSON(layer);
   std::cout << "\n\n" << new_json << std::endl;
}

#endif //TEST_PURELIN_ACTIVATION_H
