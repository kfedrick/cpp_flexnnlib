//
// Created by kfedrick on 2/27/21.
//

#include "FaninNNActivationTestFixture.h"
#include "Reinforcement.h"

#include <fstream>
#include <rapidjson/istreamwrapper.h>


using flexnnet::PureLin;
using flexnnet::BaseNeuralNet;
using flexnnet::NetworkLayerImpl;
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
   const rapidjson::Value& test_cases_arr = doc["activation_test_cases"].GetArray();
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

template<typename T> void FaninNNActivationTestFixture<T>::create_newnnet(const TestCase& _testcase)
{
   // Create network topology
   flexnnet::NeuralNetTopology topo;

   std::shared_ptr<NetworkLayerImpl<T>> hl1_ptr = std::make_shared<NetworkLayerImpl<T>>(NetworkLayerImpl<T>(_testcase.hlayer1_sz, "hidden1", T::DEFAULT_PARAMS, false));

   std::shared_ptr<NetworkLayerImpl<T>> hl2_ptr = std::make_shared<NetworkLayerImpl<T>>(NetworkLayerImpl<T>(_testcase.hlayer2_sz, "hidden2", T::DEFAULT_PARAMS, false));

   size_t total_sz = _testcase.hlayer1_sz + _testcase.hlayer2_sz;
   std::shared_ptr<NetworkLayerImpl<PureLin>> ol_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(total_sz, "output", PureLin::DEFAULT_PARAMS, true));

   hl1_ptr->add_external_input_field("input1", _testcase.hlayer1_input_sz);
   hl2_ptr->add_external_input_field("input2", _testcase.hlayer2_input_sz);
   ol_ptr->add_connection("activation", hl1_ptr, LayerConnRecord::Forward);
   ol_ptr->add_connection("activation", hl2_ptr, LayerConnRecord::Forward);
   hl1_ptr->add_connection("backprop", ol_ptr, LayerConnRecord::Forward);
   hl2_ptr->add_connection("backprop", ol_ptr, LayerConnRecord::Forward);

   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_layers[hl1_ptr->name()] = hl1_ptr;
   topo.network_layers[hl2_ptr->name()] = hl2_ptr;

   topo.network_output_layers.push_back(ol_ptr);

   topo.ordered_layers.push_back(hl1_ptr);
   topo.ordered_layers.push_back(hl2_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   newnnet = std::make_shared<BaseNeuralNet>(BaseNeuralNet(topo));

   // Set network weights
   newnnet->set_weights("output", _testcase.output_weights);
   newnnet->set_weights("hidden1", _testcase.hlayer1_weights);
   newnnet->set_weights("hidden2", _testcase.hlayer2_weights);
}




TYPED_TEST_P (FaninNNActivationTestFixture, NNActivationTest)
{
   std::vector<typename FaninNNActivationTestFixture<TypeParam>::TestCase> test_samples;

   // Get parameterized type string
   std::string layer_type_id = FaninNNActivationTestFixture<TypeParam>::get_typeid();
   std::cout << "\nTest New Fan-in Hidden Units Network<" << layer_type_id << ">\n";

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   // Set file name containing test cases
   std::string sample_fname = "fanin_" + _id + "_nnet_test_cases.json";

   std::vector<typename FaninNNActivationTestFixture<TypeParam>::TestCase>
      test_cases = FaninNNActivationTestFixture<
      TypeParam>::read_samples(sample_fname);
   for (auto test_case : test_cases)
   {
      // Create the neural network
      this->create_newnnet(test_case);

      // Print out the input vector
      std::cout << this->prettyPrintVector("input1", test_case.input.at("input1")).c_str() << "\n" << std::flush;
      std::cout << this->prettyPrintVector("input2", test_case.input.at("input2")).c_str() << "\n" << std::flush;

      // Activate the network
      const flexnnet::ValarrMap& netout = this->newnnet->activate(test_case.input);

      // Print out the network output
      std::cout << this->prettyPrintVector("netout", netout.at("output")).c_str() << "\n" << std::flush;

      // Check layer output
      EXPECT_PRED3(CommonTestFixtureFunctions::valarray_double_near, test_case.target_output.at("output"), netout.at("output"), 0.000000001) << "ruh roh";

      std::cout << "----- Done with test " << _id.c_str() << " ----\n";
      std::flush(std::cout);
   }
}

REGISTER_TYPED_TEST_CASE_P
(FaninNNActivationTestFixture, NNActivationTest);
INSTANTIATE_TYPED_TEST_CASE_P
(My, FaninNNActivationTestFixture, MyTypes);
