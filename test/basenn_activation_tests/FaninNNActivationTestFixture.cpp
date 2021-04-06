//
// Created by kfedrick on 2/27/21.
//

#include "FaninNNActivationTestFixture.h"

#include <fstream>
#include <rapidjson/istreamwrapper.h>

#include "NetworkTopology.h"
#include "BaseNeuralNet.h"

using flexnnet::NetworkTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::PureLin;
using flexnnet::LayerConnRecord;

template<typename T> std::vector<typename FaninNNActivationTestFixture<T>::TestCase>
FaninNNActivationTestFixture<T>::read_samples(std::string _filepath)
{
   std::vector<TestCase> test_samples;
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
      TestCase test_case;

      test_case.hlayer1_sz = doc["hlayer1_size"].GetUint64();
      test_case.hlayer1_input_sz = doc["hlayer1_input_size"].GetUint64();
      test_case.hlayer2_sz = doc["hlayer2_size"].GetUint64();
      test_case.hlayer2_input_sz = doc["hlayer2_input_size"].GetUint64();

      static std::valarray<double> input1v(test_case.hlayer1_input_sz);
      static std::valarray<double> input2v(test_case.hlayer2_input_sz);

//      size_t total_sz = test_case.hlayer1_sz + test_case.hlayer2_sz;
//      static std::valarray<double> outputv(total_sz);

      // Read hidden basic_layer #1 weights
      const rapidjson::Value& weights_obj1 = a_tuple_obj["hlayer1_weights"];
      test_case.hlayer1_weights.resize(test_case.hlayer1_sz, test_case.hlayer1_input_sz + 1);
      test_case.hlayer1_weights = parse_weights(weights_obj1, test_case.hlayer1_sz, test_case.hlayer1_input_sz + 1);

      // Read hidden basic_layer #1 weights
      const rapidjson::Value& weights_obj2 = a_tuple_obj["hlayer2_weights"];
      test_case.hlayer2_weights.resize(test_case.hlayer2_sz, test_case.hlayer2_input_sz + 1);
      test_case.hlayer2_weights = parse_weights(weights_obj2, test_case.hlayer2_sz, test_case.hlayer2_input_sz + 1);

      // Read output layer weights
      size_t total_sz = test_case.hlayer1_sz + test_case.hlayer2_sz;
      const rapidjson::Value& weights_obj3 = a_tuple_obj["output_weights"];
      test_case.output_weights.resize(total_sz, total_sz+1);
      test_case.output_weights = parse_weights(weights_obj3, total_sz, total_sz+1);

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

template<typename T> void FaninNNActivationTestFixture<T>::create_nnet(const TestCase& _testcase)
{
   // Set network basic_layer names
/*   std::string type_id = FaninNNActivationTestFixture<T>::get_typeid();
   std::string layer_id = type_id;
   std::transform(layer_id.begin(), layer_id.end(), layer_id.begin(), ::tolower);
   std::set<std::string> layerid_set = {layer_id};*/

   // Create network topology
   NetworkTopology nettopo(_testcase.input);

   nettopo.add_layer<T>("hidden1", _testcase.hlayer1_sz, false);
   nettopo.add_external_input_field("hidden1", "input1");

   nettopo.add_layer<T>("hidden2", _testcase.hlayer2_sz, false);
   nettopo.add_external_input_field("hidden2", "input2");

   size_t total_sz = _testcase.hlayer1_sz + _testcase.hlayer2_sz;
   nettopo.add_layer<PureLin>("output", total_sz, true);

   // Add input from both hidden basic_layer
   nettopo.add_layer_connection("output", "hidden1", LayerConnRecord::Forward);
   nettopo.add_layer_connection("output", "hidden2", LayerConnRecord::Forward);

   // Create neural net
   BaseNeuralNet nn(nettopo);
   nnet = std::shared_ptr<BaseNeuralNet>(new BaseNeuralNet(nettopo));

   // Set network weights
   nnet->set_weights(
      {
         {"hidden1", _testcase.hlayer1_weights},
         {"hidden2", _testcase.hlayer2_weights},
         {"output", _testcase.output_weights}
      });
}

TYPED_TEST_P (FaninNNActivationTestFixture, ReadTestCase)
{
   std::vector<typename FaninNNActivationTestFixture<TypeParam>::TestCase> test_samples;

   // Get parameterized type string
   std::string layer_type_id = FaninNNActivationTestFixture<TypeParam>::get_typeid();
   std::cout << "\nTest Fan-in Hidden Units Network<" << layer_type_id << ">\n";

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   // Set file name containing test cases
   std::string sample_fname = "fanin_" + _id + "_nnet_test_cases.json";

   std::vector<typename FaninNNActivationTestFixture<TypeParam>::TestCase> test_cases = FaninNNActivationTestFixture<
      TypeParam>::read_samples(sample_fname);
   for (auto test_case : test_cases)
   {
      // Create the neural network
      this->create_nnet(test_case);

      // Print out the input vector
      std::cout << this->prettyPrintVector("input1", test_case.input.at("input1")).c_str() << "\n" << std::flush;
      std::cout << this->prettyPrintVector("input2", test_case.input.at("input2")).c_str() << "\n" << std::flush;

      // Activate the network
      const std::valarray<double>& netout = this->nnet->activate(test_case.input);

      // Print out the network output
      std::cout << this->prettyPrintVector("netout", netout).c_str() << "\n";

      // Check layer output
      EXPECT_PRED3(CommonTestFixtureFunctions::valarray_double_near, test_case.target_output.at("output"), netout, 0.000000001) << "ruh roh";

      std::cout << "----- Done with test " << _id.c_str() << " ----\n";
      std::flush(std::cout);
   }
}

/*REGISTER_TYPED_TEST_CASE_P
(FaninNNActivationTestFixture, ReadTestCase);
INSTANTIATE_TYPED_TEST_CASE_P
(My, FaninNNActivationTestFixture, MyTypes);*/
