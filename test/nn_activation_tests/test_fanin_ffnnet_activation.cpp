//
// Created by kfedrick on 6/27/19.
//

#include "test_fanin_ffnnet_activation.h"

#include <iostream>
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>

#include "TestLayer.h"
#include "OldNetworkLayer.h"

using flexnnet::Array2D;
using flexnnet::Datum;
using flexnnet::OldNetworkLayer;
using flexnnet::BasicNeuralNet;

static bool
array_double_near(const flexnnet::Array2D<double>& _target, const flexnnet::Array2D<double>& _test, double _epsilon);

template<typename T>
Array2D<double> TestFaninFFNNActivation<T>::parse_weights(const rapidjson::Value& _obj, size_t _rows, size_t _cols)
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

template<typename T> NNetIO_Typ TestFaninFFNNActivation<T>::parse_datum(const rapidjson::Value& _obj)
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

template<typename T> std::vector<FaninTestCase> TestFaninFFNNActivation<T>::read_samples(std::string _filepath)
{
   FaninTestCase atest_case;

   std::vector<FaninTestCase> test_samples;

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
      FaninTestCase test_case;

      test_case.hlayer1_sz = doc["hlayer1_size"].GetUint64();
      test_case.hlayer1_input_sz = doc["hlayer1_input_size"].GetUint64();
      test_case.hlayer2_sz = doc["hlayer2_size"].GetUint64();
      test_case.hlayer2_input_sz = doc["hlayer2_input_size"].GetUint64();

      static std::valarray<double> input1v(test_case.hlayer1_input_sz);
      static std::valarray<double> input2v(test_case.hlayer2_input_sz);

      size_t total_sz = test_case.hlayer1_sz + test_case.hlayer2_sz;
      static std::valarray<double> outputv(total_sz);

      // Read hidden basic_layer #1 weights
      const rapidjson::Value& weights_obj1 = a_tuple_obj["hlayer1_weights"];
      test_case.hlayer1_weights.resize(test_case.hlayer1_sz, test_case.hlayer1_input_sz + 1);
      test_case.hlayer1_weights = parse_weights(weights_obj1, test_case.hlayer1_sz, test_case.hlayer1_input_sz + 1);

      // Read hidden basic_layer #1 weights
      const rapidjson::Value& weights_obj2 = a_tuple_obj["hlayer2_weights"];
      test_case.hlayer2_weights.resize(test_case.hlayer2_sz, test_case.hlayer2_input_sz + 1);
      test_case.hlayer2_weights = parse_weights(weights_obj2, test_case.hlayer2_sz, test_case.hlayer2_input_sz + 1);

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

TYPED_TEST_P (TestFaninFFNNActivation, FaninFFNNActivation)
{

   // Set network basic_layer names
   std::string layer_type_id = TestFaninFFNNActivation<TypeParam>::get_typeid();

   std::string layer_id = layer_type_id;
   std::transform(layer_id.begin(), layer_id.end(), layer_id.begin(), ::tolower);

   std::string sample_fname = "fanin_" + layer_id + "_ffnnet_test_cases.json";

   std::cout << "\nTest Fanin Feedforward Network<" << layer_type_id << ">\n";

   std::vector<FaninTestCase> test_cases = TestFaninFFNNActivation<TypeParam>::read_samples(sample_fname);

   for (auto test_case : test_cases)
   {
      TestFaninFFNNActivation<TypeParam>::create_fanin_ffnnet(test_case);

      Datum netout = TestFaninFFNNActivation<TypeParam>::nnet->activate(test_case.input);

      // Check basic_layer output
      EXPECT_PRED3(TestLayer::datum_near, test_case.target_output, netout, 0.000000001) << "ruh roh";

      std::cout.setf(std::ios::fixed, std::ios::floatfield);
      std::cout.precision(10);
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

TYPED_TEST_P (TestFaninFFNNActivation, FaninSaveWeights)
{
   std::cout << ".... SaveNetworkWeights Test\n";

   // Set network basic_layer names
   std::string layer_type_id = TestFaninFFNNActivation<TypeParam>::get_typeid();

   std::string layer_id = layer_type_id;
   std::transform(layer_id.begin(), layer_id.end(), layer_id.begin(), ::tolower);

   std::string sample_fname = "fanin_" + layer_id + "_ffnnet_test_cases.json";

   std::cout << "\nTest Fanin Feedforward Network<" << layer_type_id << ">\n";

   std::vector<FaninTestCase> test_cases = TestFaninFFNNActivation<TypeParam>::read_samples(sample_fname);

   for (auto test_case : test_cases)
   {
      TestFaninFFNNActivation<TypeParam>::create_fanin_ffnnet(test_case);

      NetworkWeights network_weights = TestFaninFFNNActivation<TypeParam>::nnet->get_weights();

      const flexnnet::LayerWeights& lweights = network_weights["input2"];

      for (size_t row = 0; row < lweights.const_weights_ref.size().rows; row++)
      {
         for (size_t col = 0; col < lweights.const_weights_ref.size().cols; col++)
            std::cout << lweights.const_weights_ref(row, col) << " ";
         std::cout << "\n";
      }
      std::cout << "\n";

      for (auto alayer_ptr : TestFaninFFNNActivation<TypeParam>::nnet->get_layers())
         alayer_ptr->layer_weights.zero();

      //net->set_weights(network_weights);

      for (auto alayer_ptr : TestFaninFFNNActivation<TypeParam>::nnet->get_layers())
      {
         const flexnnet::Array2D<double>& item = alayer_ptr->layer_weights.const_weights_ref;
         const flexnnet::Array2D<double>& target = network_weights[alayer_ptr->name()].const_weights_ref;
         //EXPECT_PRED3(array_double_near, item, target, 0.000000001) << "ruh roh";
      }
   }
}

REGISTER_TYPED_TEST_CASE_P
(TestFaninFFNNActivation, FaninFFNNActivation, FaninSaveWeights);
INSTANTIATE_TYPED_TEST_CASE_P
(My, TestFaninFFNNActivation, MyTypes);


