//
// Created by kfedrick on 5/19/19.
//

#ifndef _LAYERACTIVATIONTESTCASE_H_
#define _LAYERACTIVATIONTESTCASE_H_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>

#include "TestLayer.h"

#include "Array2D.h"
#include "LayerSerializer.h"

#include "PureLin.h"
#include "LogSig.h"

template<class _LayerType> class LayerActivationTestCase : public TestLayer
{
public:
   virtual void SetUp ()
   {}
   virtual void TearDown ()
   {}

   /**
    * Test case includes layer input and initial conditions and
    * expected values for layer output and derivatives for the
    * specified inputs.
    */
   struct TestCaseRecord
   {
      std::valarray<double> initial_value;
      std::valarray<double> input;

      struct Target
      {
         std::valarray<double> output;
         flexnnet::Array2D<double> dAdN;
         flexnnet::Array2D<double> dNdW;
         flexnnet::Array2D<double> dNdI;
      };

      Target target;
   };

public:
   void read (const std::string &_filepath);

public:
   void readTestCases ();

public:
   std::shared_ptr<_LayerType> layer_ptr;
   std::vector<TestCaseRecord> samples;

private:
   rapidjson::Document doc;
};

class TestPureLinActivation : public TestLayer, public ::testing::TestWithParam<const char *>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestPureLinActivation, ::testing::Values ("purelin_activity_test1.json", "purelin_activity_test2.json"));


class TestLogSigActivation : public TestLayer, public ::testing::TestWithParam<const char *>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestLogSigActivation, ::testing::Values ("logsig_activity_test1.json", "logsig_activity_test2.json"));

class TestTanSigActivation : public TestLayer, public ::testing::TestWithParam<const char *>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestTanSigActivation, ::testing::Values ("tansig_activity_test1.json", "tansig_activity_test2.json"));


class TestRadBasActivation : public TestLayer, public ::testing::TestWithParam<const char *>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestRadBasActivation, ::testing::Values ("radbas_activity_test1.json"));

class TestSoftMaxActivation : public TestLayer, public ::testing::TestWithParam<const char *>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestSoftMaxActivation, ::testing::Values ("softmax_activity_test1.json"));

template<class _LayerType> void LayerActivationTestCase<_LayerType>::read (const std::string &_filepath)
{
   std::cout << "\n" << _filepath << "\n";

   // Open file and create rabidjson file stream wrapper
   std::ifstream in_fstream (_filepath);
   rapidjson::IStreamWrapper in_fswrapper (in_fstream);

   // Parse json file stream into rapidjson document
   doc.ParseStream (in_fswrapper);

   //Save layer identifier information
   rapidjson::Value layer_def_obj = doc["layer_definition"].GetObject ();

   rapidjson::StringBuffer strbuf;
   rapidjson::Writer<rapidjson::StringBuffer> writer (strbuf);
   layer_def_obj.Accept (writer);
   std::string layer_def_str = strbuf.GetString ();

   std::cout << "\n" << layer_def_str << "\n";

   layer_ptr = flexnnet::LayerSerializer<_LayerType>::parse (layer_def_str);

   readTestCases ();

   in_fstream.close ();
}

template<class _LayerType> void LayerActivationTestCase<_LayerType>::readTestCases ()
{
   size_t layer_size = layer_ptr->size ();
   size_t input_size = layer_ptr->input_size ();

   const rapidjson::Value &test_cases_arr = doc["test_cases"].GetArray ();

   /*
    * Iterate through all of the test pairs in the "test_pairs" vector
    */
   for (rapidjson::SizeType i = 0; i < test_cases_arr.Size (); i++)
   {
      // save a reference to the i'th test pair
      const rapidjson::Value &a_tuple_obj = test_cases_arr[i];

      // Set the input vector to the correct size and copy the sample input vector
      TestCaseRecord test_sample;

      test_sample.input.resize (input_size);
      test_sample.initial_value.resize (layer_size);
      test_sample.target.output.resize (layer_size);

      // Read the test input vector
      const rapidjson::Value &in_arr = a_tuple_obj["input"];
      for (rapidjson::SizeType i = 0; i < in_arr.Size (); i++)
         test_sample.input[i] = in_arr[i].GetDouble ();

      // Read the test input vector
      const rapidjson::Value &init_arr = a_tuple_obj["initial_value"];
      for (rapidjson::SizeType i = 0; i < init_arr.Size (); i++)
         test_sample.initial_value[i] = init_arr[i].GetDouble ();

      // Read the test input vector
      const rapidjson::Value &out_arr = a_tuple_obj["target"]["output"];
      for (rapidjson::SizeType i = 0; i < out_arr.Size (); i++)
         test_sample.target.output[i] = out_arr[i].GetDouble ();

      // Read dAdN
      const rapidjson::Value &dAdN_arr = a_tuple_obj["target"]["dAdN"];
      test_sample.target.dAdN.resize (layer_size, layer_size);
      for (rapidjson::SizeType i = 0; i < dAdN_arr.Size (); i++)
      {
         const rapidjson::Value &myrow = dAdN_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size (); j++)
            test_sample.target.dAdN.at(i, j) = myrow[j].GetDouble ();
      }

      // Read dNdW
      const rapidjson::Value &dNdW_arr = a_tuple_obj["target"]["dNdW"];
      test_sample.target.dNdW.resize (layer_size, input_size + 1);
      for (rapidjson::SizeType i = 0; i < dNdW_arr.Size (); i++)
      {
         const rapidjson::Value &myrow = dNdW_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size (); j++)
            test_sample.target.dNdW.at(i, j) = myrow[j].GetDouble ();
      }

      // Read dNdI
      const rapidjson::Value &dNdI_arr = a_tuple_obj["target"]["dNdI"];
      test_sample.target.dNdI.resize (layer_size, input_size + 1);
      for (rapidjson::SizeType i = 0; i < dNdI_arr.Size (); i++)
      {
         const rapidjson::Value &myrow = dNdI_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size (); j++)
            test_sample.target.dNdI.at(i, j) = myrow[j].GetDouble ();
      }

      // Push a copy of the test pair onto the test pairs vector
      samples.push_back (test_sample);
   }
}

#endif //_LAYERACTIVATIONTESTCASE_H_
