//
// Created by kfedrick on 2/22/21.
//
#include "SingleLayerNNActivationTestFixture.h"

#include <fstream>
#include <rapidjson/istreamwrapper.h>

#include "NetworkTopology.h"
#include "BaseNeuralNet.h"

using flexnnet::NetworkTopology;
using flexnnet::BaseNeuralNet;

template<typename T> std::vector<typename SingleLayerNNActivationTestFixture<T>::TestCase> SingleLayerNNActivationTestFixture<T>::read_samples(std::string _filepath)
{
   std::vector<TestCase> test_samples;

   std::cout << "\n" << _filepath << "\n";

   // Open file and create rabidjson file stream wrapper
   std::ifstream in_fstream(_filepath);
   rapidjson::IStreamWrapper in_fswrapper(in_fstream);


   // Parse json file stream into rapidjson document
   rapidjson::Document doc;
   doc.ParseStream(in_fswrapper);

   size_t layer_sz = doc["layer_size"].GetInt();
   size_t input_sz = doc["input_size"].GetInt();

   // Iterate through test input/output pairs
   const rapidjson::Value& test_cases_arr = doc["test_cases"].GetArray();
   for (rapidjson::SizeType i = 0; i < test_cases_arr.Size(); i++)
   {
      static std::valarray<double> inputv(input_sz);
      static std::valarray<double> outputv(layer_sz);

      // save a reference to the i'th test pair
      const rapidjson::Value& a_tuple_obj = test_cases_arr[i];

      // Set the input vector to the correct size and copy the sample input vector
      TestCase test_case;

      test_case.layer_sz = layer_sz;
      test_case.input_sz = input_sz;

      // Read basic_layer weights
      const rapidjson::Value& weights_obj = a_tuple_obj["weights"];
      test_case.weights.resize(layer_sz, input_sz + 1);
      test_case.weights = parse_weights(weights_obj, layer_sz, input_sz + 1);

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

template<typename T> void SingleLayerNNActivationTestFixture<T>::create_nnet(const TestCase& _testcase)
{
   // Set network basic_layer names
   std::string type_id = SingleLayerNNActivationTestFixture<T>::get_typeid();
   std::string layer_id = type_id;
   std::transform(layer_id.begin(), layer_id.end(), layer_id.begin(), ::tolower);
   std::set<std::string> layerid_set = {layer_id};

   // Create network topology
   NetworkTopology nettopo(_testcase.input);

   nettopo.add_layer<T>(layer_id, _testcase.layer_sz, true);
   nettopo.add_external_input_field(layer_id, "input");

   nnet = std::shared_ptr<BaseNeuralNet>(new BaseNeuralNet(nettopo));

   // Set network weights
   nnet->set_weights({{layer_id, _testcase.weights}});
}

TYPED_TEST_P (SingleLayerNNActivationTestFixture, ActivationTest)
{

   std::vector<typename SingleLayerNNActivationTestFixture<TypeParam>::TestCase> test_samples;

   // Get parameterized type string
   std::string layer_type_id = SingleLayerNNActivationTestFixture<TypeParam>::get_typeid();
   std::cout << "\nTest Single Layer Network<" << layer_type_id << ">\n";

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   // Set file name containing test cases
   std::string sample_fname = "single_" + _id + "_nnet_test_cases.json";

   std::vector<typename SingleLayerNNActivationTestFixture<TypeParam>::TestCase> test_cases = SingleLayerNNActivationTestFixture<TypeParam>::read_samples(sample_fname);
   for (auto test_case : test_cases)
   {
      // Create the neural network
      this->create_nnet(test_case);

      // Print out the input vector
      std::cout << this->prettyPrintVector("input", test_case.input.at("input")).c_str() << "\n";
      std::flush(std::cout);

      // Activate the network
      const std::valarray<double>& netout = this->nnet->activate(test_case.input);

      // Print out the network output
      std::cout << this->prettyPrintVector("netout", netout).c_str() << "\n";

      // Check layer output
      EXPECT_PRED3(CommonTestFixtureFunctions::valarray_double_near, test_case.target_output.at(_id), netout, 0.000000001) << "ruh roh";

      std::cout << "----- Done with test " << _id.c_str() << " ----\n";
      std::flush(std::cout);
   }
}

REGISTER_TYPED_TEST_CASE_P
(SingleLayerNNActivationTestFixture, ActivationTest);
INSTANTIATE_TYPED_TEST_CASE_P
(My, SingleLayerNNActivationTestFixture, MyTypes);