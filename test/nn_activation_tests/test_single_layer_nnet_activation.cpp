//
// Created by kfedrick on 6/25/19.
//

#include "test_single_layer_nnet_activation.h"
#include "TestLayer.h"

#include <iostream>
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>

#include "TestLayer.h"
#include "NetworkLayer.h"

using flexnnet::Array2D;
using flexnnet::Datum;
using flexnnet::NetworkLayer;
using flexnnet::BasicNeuralNet;


template<typename T> Array2D<double> TestSingleLayerNNActivation<T>::parse_weights (const rapidjson::Value &_obj, size_t _rows, size_t _cols)
{
   Array2D<double> weights(_rows, _cols);

   for (rapidjson::SizeType i=0; i<_obj.Size(); i++)
   {
      const rapidjson::Value& myrow = _obj[i];

      for (rapidjson::SizeType j=0; j<myrow.Size(); j++)
         weights.at(i, j) = myrow[j].GetDouble();
   }

   return weights;
}

template<typename T> Datum TestSingleLayerNNActivation<T>::parse_datum (const rapidjson::Value &_obj)
{
   std::map< std::string, std::valarray<double> > datum_fields;
   for (rapidjson::SizeType i=0; i<_obj.Size(); i++)
   {
      std::string field = _obj[i]["field"].GetString();
      size_t field_sz = _obj[i]["size"].GetUint64 ();
      size_t field_index = _obj[i]["index"].GetUint64 ();

      datum_fields[field] = std::valarray<double>(field_sz);

      const rapidjson::Value& vec = _obj[i]["value"];
      for (rapidjson::SizeType i = 0; i < vec.Size (); i++)
         datum_fields[field][i] = vec[i].GetDouble();
   }

   Datum datum(datum_fields);
   return datum;
}

template<typename T> std::vector<TestCase> TestSingleLayerNNActivation<T>::read_samples(std::string _filepath)
{
   std::vector<TestCase> test_samples;

   std::cout << "\n" << _filepath << "\n";

   // Open file and create rabidjson file stream wrapper
   std::ifstream in_fstream (_filepath);
   rapidjson::IStreamWrapper in_fswrapper (in_fstream);

   // Parse json file stream into rapidjson document
   rapidjson::Document doc;
   doc.ParseStream (in_fswrapper);

   size_t layer_sz = doc["layer_size"].GetDouble();
   size_t input_sz = doc["input_size"].GetDouble();

   // Iterate through test input/output pairs
   const rapidjson::Value &test_cases_arr = doc["test_cases"].GetArray ();
   for (rapidjson::SizeType i = 0; i < test_cases_arr.Size (); i++)
   {
      static std::valarray<double> inputv(input_sz);
      static std::valarray<double> outputv(layer_sz);

      // save a reference to the i'th test pair
      const rapidjson::Value &a_tuple_obj = test_cases_arr[i];

      // Set the input vector to the correct size and copy the sample input vector
      TestCase test_case;

      test_case.layer_sz = layer_sz;
      test_case.input_sz = input_sz;

      // Read layer weights
      const rapidjson::Value &weights_obj = a_tuple_obj["weights"];
      test_case.weights.resize(layer_sz, input_sz+1);
      test_case.weights = parse_weights (weights_obj, layer_sz, input_sz+1);

      // Read the test input vector
      const rapidjson::Value &indatum_obj = a_tuple_obj["input"];
      test_case.input = parse_datum (indatum_obj);

      // Read the test input vector
      const rapidjson::Value &outdatum_obj = a_tuple_obj["output"];
      test_case.target_output = parse_datum (outdatum_obj);

      test_samples.push_back(test_case);
   }

   return test_samples;
}



TYPED_TEST_P (TestSingleLayerNNActivation, SingleLayerNNActivation)
{
   // Set network layer names
   std::string layer_type_id = TestSingleLayerNNActivation<TypeParam>::get_typeid();

   std::string layer_id = layer_type_id;
   std::transform(layer_id.begin(), layer_id.end(), layer_id.begin(), ::tolower);

   std::string sample_fname = "single_" + layer_id + "_nnet_test_cases.json";

   std::cout << "\nTest Single Layer Network<" << layer_type_id << ">\n";

   std::vector<TestCase> test_cases = TestSingleLayerNNActivation<TypeParam>::read_samples(sample_fname);

   for (auto test_case : test_cases)
   {
      TestSingleLayerNNActivation<TypeParam>::create_single_layer_nnet (test_case);
      Datum netout = TestSingleLayerNNActivation<TypeParam>::nnet->activate(test_case.input);

      // Check layer output
      EXPECT_PRED3(TestLayer::datum_near, test_case.target_output, netout, 0.000000001) << "ruh roh";

      std::cout.setf (std::ios::fixed, std::ios::floatfield);
      std::cout.precision (10);
      std::cout << "\nnetwork output : \n{";
      bool first = true;
      for (auto val : netout())
      {
         if (!first)
            std::cout << ", ";

         std::cout << val;
         first = false;
      }
      std::cout << "}\n";

      std::cout << "\ntarget output : \n{";
      first = true;
      for (auto val : test_case.target_output())
      {
         if (!first)
            std::cout << ", ";

         std::cout << val;
         first = false;
      }
      std::cout << "}\n";
   }
}

TYPED_TEST_P (TestSingleLayerNNActivation, TwoLayerFFNActivation)
{
   // Set network layer names
   std::string layer_type_id = TestSingleLayerNNActivation<TypeParam>::get_typeid();

   std::string layer_id = layer_type_id;
   std::transform(layer_id.begin(), layer_id.end(), layer_id.begin(), ::tolower);

   std::string sample_fname = "two_layer_" + layer_id + "_ffnnet_test_cases.json";

   std::cout << "\nTest Two Layer Feedforward Network<" << layer_type_id << ">\n";

   std::vector<TestCase> test_cases = TestSingleLayerNNActivation<TypeParam>::read_samples(sample_fname);

   for (auto test_case : test_cases)
   {
      TestSingleLayerNNActivation<TypeParam>::create_two_layer_ffnnet (test_case);
      Datum netout = TestSingleLayerNNActivation<TypeParam>::nnet->activate(test_case.input);

      // Check layer output
      EXPECT_PRED3(TestLayer::datum_near, test_case.target_output, netout, 0.000000001) << "ruh roh";

      std::cout.setf (std::ios::fixed, std::ios::floatfield);
      std::cout.precision (10);
      std::cout << "\nnetwork output : \n{";
      bool first = true;
      for (auto val : netout())
      {
         if (!first)
            std::cout << ", ";

         std::cout << val;
         first = false;
      }
      std::cout << "}\n";
   }
}

REGISTER_TYPED_TEST_CASE_P(TestSingleLayerNNActivation, SingleLayerNNActivation, TwoLayerFFNActivation);
INSTANTIATE_TYPED_TEST_CASE_P(My, TestSingleLayerNNActivation, MyTypes);