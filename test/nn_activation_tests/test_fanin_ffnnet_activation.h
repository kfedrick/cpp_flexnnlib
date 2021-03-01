//
// Created by kfedrick on 6/27/19.
//

#ifndef _TEST_FANIN_FFNNET_ACTIVATION_H_
#define _TEST_FANIN_FFNNET_ACTIVATION_H_

#include <gtest/gtest.h>

#include <string>
#include <valarray>

#include "Array2D.h"
#include "BasicLayer.h"
#include "PureLin.h"
#include "TanSig.h"
#include "RadBas.h"
#include "SoftMax.h"
#include "LogSig.h"

#include "TestLayer.h"

using flexnnet::NNetIO_Typ;

#define TESTCASE_PATH "test/nn_activation_tests/samples/"

struct FaninTestCase
{
   size_t hlayer1_sz;
   size_t hlayer1_input_sz;

   size_t hlayer2_sz;
   size_t hlayer2_input_sz;

   flexnnet::Array2D<double> hlayer1_weights;
   int wtf;
   flexnnet::Array2D<double> hlayer2_weights;

   NNetIO_Typ input;
   NNetIO_Typ target_output;
};

template<typename T>
class TestFaninFFNNActivation : public TestLayer, public ::testing::Test
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   std::string get_typeid();
   void create_fanin_ffnnet(const FaninTestCase& _testcase);

   std::vector<FaninTestCase> read_samples(std::string _fpath);
   NNetIO_Typ parse_datum(const rapidjson::Value& _obj);
   flexnnet::Array2D<double> parse_weights(const rapidjson::Value& _obj, size_t _rows, size_t _cols);

public:

   std::string HIDDEN_LAYER_TYPE_ID;
   std::string FANIN_LAYER1_ID;
   std::string FANIN_LAYER2_ID;

   std::set<std::string> LAYER_IDS;

   std::shared_ptr<flexnnet::BasicNeuralNet> nnet;
};

TYPED_TEST_CASE_P
(TestFaninFFNNActivation);

typedef ::testing::Types<flexnnet::PureLin,
                         flexnnet::TanSig,
                         flexnnet::LogSig,
                         flexnnet::RadBas,
                         flexnnet::SoftMax> MyTypes;

template<typename T> std::string TestFaninFFNNActivation<T>::get_typeid()
{
   std::string type_id = typeid(T).name();
   static char buf[2048];
   size_t size = sizeof(buf);
   int status;

   char* res = abi::__cxa_demangle(type_id.c_str(), buf, &size, &status);
   buf[sizeof(buf) - 1] = 0;

   std::string buf_str = buf;
   size_t pos = buf_str.rfind(':') + 1;
   buf_str = buf_str.substr(pos);

   return buf_str;
}

template<typename T> void TestFaninFFNNActivation<T>::create_fanin_ffnnet(const FaninTestCase& _testcase)
{
   // Set network basic_layer names
   HIDDEN_LAYER_TYPE_ID = TestFaninFFNNActivation<T>::get_typeid();
   FANIN_LAYER1_ID = HIDDEN_LAYER_TYPE_ID + "1";
   FANIN_LAYER2_ID = HIDDEN_LAYER_TYPE_ID + "2";

   std::transform(FANIN_LAYER2_ID.begin(), FANIN_LAYER2_ID.end(), FANIN_LAYER2_ID.begin(), ::tolower);
   std::transform(FANIN_LAYER1_ID.begin(), FANIN_LAYER1_ID.end(), FANIN_LAYER1_ID.begin(), ::tolower);
   LAYER_IDS = {FANIN_LAYER1_ID, FANIN_LAYER2_ID, "output"};

   std::vector<std::shared_ptr<flexnnet::OldNetworkLayer>> network_layers;

   // Create hidden fanin basic_layer #1
   network_layers
      .push_back(std::shared_ptr<T>(new T(_testcase.hlayer1_sz, FANIN_LAYER1_ID, flexnnet::OldNetworkLayer::Hidden)));

   // Add external input to hidden fanin basic_layer #1
   network_layers[0]->add_external_input(_testcase.input, {"input1"});

   // Set hidden fanin basic_layer #1 weights
   network_layers[0]->layer_weights.set(_testcase.hlayer1_weights);

   // Create hidden fanin basic_layer #2
   network_layers
      .push_back(std::shared_ptr<T>(new T(_testcase.hlayer2_sz, FANIN_LAYER2_ID, flexnnet::OldNetworkLayer::Hidden)));

   // Add external input to hidden fanin basic_layer #2
   network_layers[1]->add_external_input(_testcase.input, {"input2"});

   // Set hidden fanin basic_layer #2 weights
   network_layers[1]->layer_weights.set(_testcase.hlayer2_weights);

   // Create output basic_layer
   size_t total_sz = _testcase.hlayer1_sz + _testcase.hlayer2_sz;
   network_layers
      .push_back(std::shared_ptr<flexnnet::PureLin>(new flexnnet::PureLin(total_sz, "output", flexnnet::OldNetworkLayer::Output)));

   // Add input from both hidden basic_layer
   network_layers[2]->add_connection(*network_layers[0], flexnnet::OldLayerConnRecord::Forward);
   network_layers[2]->add_connection(*network_layers[1], flexnnet::OldLayerConnRecord::Forward);

   // Create diagonal matrix for output basic_layer weights
   flexnnet::Array2D<double> diag(total_sz, total_sz + 1);
   for (int i = 0; i < total_sz; i++)
      diag.at(i, i) = 1.0;
   network_layers[2]->layer_weights.set(diag);

   // Create neural net
   nnet = std::shared_ptr<flexnnet::BasicNeuralNet>(new flexnnet::BasicNeuralNet(network_layers, false, "nnet"));
}

#endif //_TEST_FANIN_FFNNET_ACTIVATION_H_
