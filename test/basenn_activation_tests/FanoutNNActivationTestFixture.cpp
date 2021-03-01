//
// Created by kfedrick on 2/28/21.
//

#include "FanoutNNActivationTestFixture.h"

#include <fstream>
#include <rapidjson/istreamwrapper.h>

#include "NetworkTopology.h"
#include "NetworkLayerImpl.h"
#include "BaseNeuralNet.h"

using flexnnet::NetworkTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::PureLin;
using flexnnet::LayerConnRecord;

template<typename T> std::vector<typename FanoutNNActivationTestFixture<T>::TestCase> FanoutNNActivationTestFixture<T>::read_samples(std::string _filepath)
{
   TestCase atest_case;

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

      test_case.hlayer_sz = doc["hlayer_size"].GetUint64();
      test_case.olayer1_sz = doc["olayer1_size"].GetUint64();
      test_case.olayer2_sz = doc["olayer2_size"].GetUint64();
      test_case.input1_sz = doc["input1_size"].GetUint64();
      test_case.input2_sz = doc["input2_size"].GetUint64();

      static std::valarray<double> input1v(test_case.input1_sz);
      static std::valarray<double> input2v(test_case.input2_sz);

      size_t total_sz = test_case.olayer1_sz + test_case.olayer2_sz;
      static std::valarray<double> outputv(total_sz);

      // Read hidden basic_layer weights
      const rapidjson::Value& weights_obj1 = a_tuple_obj["hlayer_weights"];
      test_case.hlayer_weights.resize(test_case.hlayer_sz, test_case.input1_sz + 1);
      test_case.hlayer_weights = parse_weights(weights_obj1, test_case.hlayer_sz, test_case.input1_sz + 1);

      // Read output basic_layer #1 weights
      const rapidjson::Value& weights_obj2 = a_tuple_obj["olayer1_weights"];
      test_case.olayer1_weights.resize(test_case.olayer1_sz, test_case.hlayer_sz + 1);
      test_case.olayer1_weights = parse_weights(weights_obj2, test_case.olayer1_sz, test_case.hlayer_sz + 1);

      // Read output basic_layer #2 weights
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

template<typename T> void FanoutNNActivationTestFixture<T>::create_nnet(const TestCase& _testcase)
{
   // Create network topology
   NetworkTopology nettopo(_testcase.input);

   nettopo.add_layer<T>("hidden", _testcase.hlayer_sz, false);
   nettopo.add_external_input_field("hidden", "input1");

   nettopo.add_layer<T>("output1", _testcase.olayer1_sz, true);
   nettopo.add_layer_connection("output1", "hidden", LayerConnRecord::Forward);

   nettopo.add_layer<T>("output2", _testcase.olayer2_sz, true);
   nettopo.add_layer_connection("output2", "hidden", LayerConnRecord::Forward);

   // Create neural net
   BaseNeuralNet nn(nettopo);
   nnet = std::shared_ptr<BaseNeuralNet>(new BaseNeuralNet(nettopo));

   // Set network weights
   nnet->set_weights(
      {
         {"hidden", _testcase.hlayer_weights},
         {"output1", _testcase.olayer1_weights},
         {"output2", _testcase.olayer2_weights}
      });
}

TYPED_TEST_P (FanoutNNActivationTestFixture, ReadTestCase)
{
   std::vector<typename FanoutNNActivationTestFixture<TypeParam>::TestCase> test_samples;

   // Get parameterized type string
   std::string layer_type_id = FanoutNNActivationTestFixture<TypeParam>::get_typeid();
   std::cout << "\nTest Fan-out Hidden Units Network<" << layer_type_id << ">\n";

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   // Set file name containing test cases
   std::string sample_fname = "fanout_" + _id + "_nnet_test_cases.json";
   std::vector<typename FanoutNNActivationTestFixture<TypeParam>::TestCase> test_cases = FanoutNNActivationTestFixture<
      TypeParam>::read_samples(sample_fname);
   for (auto test_case : test_cases)
   {
      // Create the neural network
      this->create_nnet(test_case);

      // Print out the input vector
      std::cout << this->prettyPrintVector("input1", test_case.input.at("input1")).c_str() << "\n" << std::flush;

      // Activate the network
      const std::valarray<double>& netout = this->nnet->activate(test_case.input);

      // Print out the network output
      const std::map<std::string, std::shared_ptr<flexnnet::NetworkLayer>> layers = this->nnet->get_layers();
      flexnnet::NNetIO_Typ outmap({{"output1", layers.at("output1")->value()}, {"output2", layers.at("output2")->value()}});
      std::cout << this->prettyPrintVector("output1", outmap.at("output1")).c_str() << "\n";
      std::cout << this->prettyPrintVector("output2", outmap.at("output2")).c_str() << "\n";

      // Check layer output
      EXPECT_PRED3(CommonTestFixtureFunctions::valarray_double_near, test_case.target_output.at("output1"), outmap.at("output1"), 0.000000001) << "ruh roh";
      EXPECT_PRED3(CommonTestFixtureFunctions::valarray_double_near, test_case.target_output.at("output2"), outmap.at("output2"), 0.000000001) << "ruh roh";

      std::cout << "----- Done with test " << _id.c_str() << " ----\n";
      std::flush(std::cout);
   }
}

REGISTER_TYPED_TEST_CASE_P
(FanoutNNActivationTestFixture, ReadTestCase);
INSTANTIATE_TYPED_TEST_CASE_P
(My, FanoutNNActivationTestFixture, MyTypes);