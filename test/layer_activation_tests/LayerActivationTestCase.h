//
// Created by kfedrick on 5/19/19.
//

#ifndef _LAYERACTIVATIONTESTCASE_H_
#define _LAYERACTIVATIONTESTCASE_H_

#include <gtest/gtest.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>
#include <CommonTestFixtureFunctions.h>

#include "Array2D.h"
#include "PureLin.h"
#include "LogSig.h"

template<class _LayerType> class LayerActivationTestCase : public CommonTestFixtureFunctions
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   /**
    * Test case includes basic_layer input and initial conditions and
    * expected values for basic_layer output and derivatives for the
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
   void read(const std::string& _filepath);

public:
   void readTestCases();

public:
   std::shared_ptr<_LayerType> layer_ptr;
   std::vector<TestCaseRecord> samples;

private:
   rapidjson::Document doc;
};

class TestPureLinActivation : public CommonTestFixtureFunctions, public ::testing::TestWithParam<const char*>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestPureLinActivation, ::testing::Values("purelin_activity_test1.json", "purelin_activity_test2.json"));

class TestLogSigActivation : public CommonTestFixtureFunctions, public ::testing::TestWithParam<const char*>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestLogSigActivation, ::testing::Values("logsig_activity_test1.json", "logsig_activity_test2.json"));

class TestTanSigActivation : public CommonTestFixtureFunctions, public ::testing::TestWithParam<const char*>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestTanSigActivation, ::testing::Values("tansig_activity_test1.json", "tansig_activity_test2.json"));

class TestRadBasActivation : public CommonTestFixtureFunctions, public ::testing::TestWithParam<const char*>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestRadBasActivation, ::testing::Values("radbas_activity_test1.json"));

class TestSoftMaxActivation : public CommonTestFixtureFunctions, public ::testing::TestWithParam<const char*>
{
};
INSTANTIATE_TEST_CASE_P
(InstantiationName, TestSoftMaxActivation, ::testing::Values("softmax_activity_test1.json"));

template<class _LayerType> void LayerActivationTestCase<_LayerType>::read(const std::string& _filepath)
{
   std::cout << "\n" << _filepath << "\n";

   // Open file and create rabidjson file stream wrapper
   std::ifstream in_fstream(_filepath);
   rapidjson::IStreamWrapper in_fswrapper(in_fstream);

   // Parse json file stream into rapidjson document
   doc.ParseStream(in_fswrapper);

   //Save basic_layer identifier information
   rapidjson::Value layer_def_obj = doc["layer_definition"].GetObject();

   std::string layer_id = layer_def_obj["id"].GetString();
   bool is_output;// = layer_def_obj["dimensions"]["is_output_layer"].GetBool();
   int layer_sz = layer_def_obj["dimensions"]["layer_size"].GetUint();
   int layer_input_sz = layer_def_obj["dimensions"]["layer_input_size"].GetUint();

   std::cout << "\nCreate layer : " << layer_id.c_str() << " " << layer_sz << "\n" << std::flush;
   layer_ptr = std::shared_ptr<_LayerType>(new _LayerType(layer_sz, layer_id));
   std::cout << "\nDone create layer\n" << std::flush;

   flexnnet::ValarrMap io({{"input", std::valarray<double>(layer_input_sz)}});
   layer_ptr->resize_input(layer_input_sz);

   // Read layer weights and set them
   const rapidjson::Value& weights_arr = layer_def_obj["learned_parameters"]["weights"];
   flexnnet::Array2D<double> weights;

   weights.resize(layer_sz, layer_input_sz + 1);
   for (rapidjson::SizeType i = 0; i < weights_arr.Size(); i++)
   {
      const rapidjson::Value& myrow = weights_arr[i];
      for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
         weights.at(i, j) = myrow[j].GetDouble();
   }
   layer_ptr->layer_weights.set(weights);

   readTestCases();
   std::cout << "\nDone reading test cases\n" << std::flush;

   in_fstream.close();
}

template<class _LayerType> void LayerActivationTestCase<_LayerType>::readTestCases()
{
   samples.clear();

   size_t layer_size = layer_ptr->size();
   size_t input_size = layer_ptr->input_size();

   std::cout << "\nLayer input size : " << input_size << "\n" << std::flush;

   const rapidjson::Value& test_cases_arr = doc["test_cases"].GetArray();

   /*
    * Iterate through all of the test pairs in the "test_pairs" vector
    */
   for (rapidjson::SizeType i = 0; i < test_cases_arr.Size(); i++)
   {
      std::cout << "\ntest case  : " << i << "\n" << std::flush;

      // save a reference to the i'th test pair
      const rapidjson::Value& a_tuple_obj = test_cases_arr[i];

      // Set the input vector to the correct size and copy the sample input vector
      TestCaseRecord test_sample;

      test_sample.input.resize(input_size);
      test_sample.initial_value.resize(layer_size);
      test_sample.target.output.resize(layer_size);

      // Read the test input vector
      const rapidjson::Value& in_arr = a_tuple_obj["input"];
      for (rapidjson::SizeType i = 0; i < in_arr.Size(); i++)
         test_sample.input[i] = in_arr[i].GetDouble();

      // Read the test input vector
      const rapidjson::Value& init_arr = a_tuple_obj["initial_value"];
      for (rapidjson::SizeType i = 0; i < init_arr.Size(); i++)
         test_sample.initial_value[i] = init_arr[i].GetDouble();

      // Read the test input vector
      const rapidjson::Value& out_arr = a_tuple_obj["target"]["output"];
      for (rapidjson::SizeType i = 0; i < out_arr.Size(); i++)
         test_sample.target.output[i] = out_arr[i].GetDouble();

      std::cout << "\ntest case  here" << "\n" << std::flush;

      // Read dy_dnet
      const rapidjson::Value& dAdN_arr = a_tuple_obj["target"]["dAdN"];
      test_sample.target.dAdN.resize(layer_size, layer_size);
      for (rapidjson::SizeType i = 0; i < dAdN_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dAdN_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
            test_sample.target.dAdN.at(i, j) = myrow[j].GetDouble();
      }

      std::cout << "\ntest case here not" << "\n" << std::flush;

      // Read dnet_dw
      const rapidjson::Value& dNdW_arr = a_tuple_obj["target"]["dNdW"];
      test_sample.target.dNdW.resize(layer_size, input_size + 1);
      for (rapidjson::SizeType i = 0; i < dNdW_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dNdW_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
            test_sample.target.dNdW.at(i, j) = myrow[j].GetDouble();
      }
      std::cout << "\ntest case here too" << "\n" << std::flush;

      // Read dnet_dx
      const rapidjson::Value& dNdI_arr = a_tuple_obj["target"]["dNdI"];
      test_sample.target.dNdI.resize(layer_size, input_size + 1);
      for (rapidjson::SizeType i = 0; i < dNdI_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dNdI_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
            test_sample.target.dNdI.at(i, j) = myrow[j].GetDouble();
      }

      // Push a copy of the test pair onto the test pairs vector
      samples.push_back(test_sample);
   }
}

#endif //_LAYERACTIVATIONTESTCASE_H_
