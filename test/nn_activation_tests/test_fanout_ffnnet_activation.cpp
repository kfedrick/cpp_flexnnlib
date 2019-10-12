//
// Created by kfedrick on 7/1/19.
//

#include "test_fanout_ffnnet_activation.h"

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

using flexnnet::NNetIO_Typ;

template<typename T>
Array2D<double> TestFanoutFFNNActivation<T>::parse_weights(const rapidjson::Value& _obj, size_t _rows, size_t _cols)
{
   Array2D<double> weights(_rows, _cols);

   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      const rapidjson::Value& myrow = _obj[i];

      for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
         weights.at(i, j) = myrow[j].GetDouble();
   }

   return weights;
}

template<typename T> NNetIO_Typ TestFanoutFFNNActivation<T>::parse_datum(const rapidjson::Value& _obj)
{
   NNetIO_Typ datum_fields;
   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      std::string field = _obj[i]["field"].GetString();
      size_t field_sz = _obj[i]["size"].GetUint64();
      size_t field_index = _obj[i]["index"].GetUint64();

      datum_fields[field] = std::valarray<double>(field_sz);

      const rapidjson::Value& vec = _obj[i]["value"];
      for (rapidjson::SizeType i = 0; i < vec.Size(); i++)
         datum_fields[field][i] = vec[i].GetDouble();
   }

   return datum_fields;
}

template<typename T> std::vector<FanoutTestCase> TestFanoutFFNNActivation<T>::read_samples(std::string _filepath)
{
   FanoutTestCase atest_case;

   std::vector<FanoutTestCase> test_samples;

   std::cout << "\n" << _filepath << "\n";

   // Open file and create rabidjson file stream wrapper
   std::ifstream in_fstream(_filepath);
   rapidjson::IStreamWrapper in_fswrapper(in_fstream);

   // Parse json file stream into rapidjson document
   rapidjson::Document doc;
   doc.ParseStream(in_fswrapper);


   // Iterate through test input/output pairs
   const rapidjson::Value& test_cases_arr = doc["test_cases"].GetArray();
   for (rapidjson::SizeType i = 0; i < test_cases_arr.Size(); i++)
   {
      // save a reference to the i'th test pair
      const rapidjson::Value& a_tuple_obj = test_cases_arr[i];

      // Set the input vector to the correct size and copy the sample input vector
      FanoutTestCase test_case;

      test_case.hlayer_sz = doc["hlayer_size"].GetUint64();
      test_case.olayer1_sz = doc["olayer1_size"].GetUint64();
      test_case.olayer2_sz = doc["olayer2_size"].GetUint64();
      test_case.input1_sz = doc["input1_size"].GetUint64();
      test_case.input2_sz = doc["input2_size"].GetUint64();

      static std::valarray<double> input1v(test_case.input1_sz);
      static std::valarray<double> input2v(test_case.input2_sz);

      size_t total_sz = test_case.olayer1_sz + test_case.olayer2_sz;
      static std::valarray<double> outputv(total_sz);

      // Read hidden layer weights
      const rapidjson::Value& weights_obj1 = a_tuple_obj["hlayer_weights"];
      test_case.hlayer_weights.resize(test_case.hlayer_sz, test_case.input1_sz + 1);
      test_case.hlayer_weights = parse_weights(weights_obj1, test_case.hlayer_sz, test_case.input1_sz + 1);

      // Read output layer #1 weights
      const rapidjson::Value& weights_obj2 = a_tuple_obj["olayer1_weights"];
      test_case.olayer1_weights.resize(test_case.olayer1_sz, test_case.hlayer_sz + 1);
      test_case.olayer1_weights = parse_weights(weights_obj2, test_case.olayer1_sz, test_case.hlayer_sz + 1);

      // Read output layer #2 weights
      const rapidjson::Value& weights_obj3 = a_tuple_obj["olayer2_weights"];
      test_case.olayer2_weights.resize(test_case.olayer2_sz, test_case.hlayer_sz + 1);
      test_case.olayer2_weights = parse_weights(weights_obj3, test_case.olayer2_sz, test_case.hlayer_sz + 1);

      // Read the test input vector
      const rapidjson::Value& indatum_obj = a_tuple_obj["input"];
      test_case.input = parse_datum(indatum_obj);

      // Read the test input vector
      const rapidjson::Value& outdatum_obj = a_tuple_obj["output"];
      test_case.target_output = parse_datum(outdatum_obj);

      test_samples.push_back(test_case);
   }

   return test_samples;
}

TYPED_TEST_P (TestFanoutFFNNActivation, FanoutFFNNActivation)
{

   // Set network layer names
   std::string layer_type_id = TestFanoutFFNNActivation<TypeParam>::get_typeid();

   std::string layer_id = layer_type_id;
   std::transform(layer_id.begin(), layer_id.end(), layer_id.begin(), ::tolower);

   std::string sample_fname = "fanout_" + layer_id + "_ffnnet_test_cases.json";

   std::cout << "\nTest Fanin Feedforward Network<" << layer_type_id << ">\n";

   std::vector<FanoutTestCase> test_cases = TestFanoutFFNNActivation<TypeParam>::read_samples(sample_fname);

   for (auto test_case : test_cases)
   {
      TestFanoutFFNNActivation<TypeParam>::create_fanout_ffnnet(test_case);

      NNetIO_Typ netout = TestFanoutFFNNActivation<TypeParam>::nnet->activate(test_case.input);

      // Check layer output
      EXPECT_PRED3(TestLayer::datum_near, test_case.target_output, netout, 0.000000001) << "ruh roh";

      std::cout.setf(std::ios::fixed, std::ios::floatfield);
      std::cout.precision(10);
      std::cout << "\nnetwork output : \n{";
      bool first = true;
      for (auto val : netout)
      {
         if (!first)
            std::cout << ", ";

         std::cout << val.first.c_str();
         first = false;
      }
      std::cout << "}\n";
   }
}

REGISTER_TYPED_TEST_CASE_P
(TestFanoutFFNNActivation, FanoutFFNNActivation);
INSTANTIATE_TYPED_TEST_CASE_P
(My, TestFanoutFFNNActivation, MyTypes);
