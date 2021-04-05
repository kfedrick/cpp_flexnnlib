//
// Created by kfedrick on 7/1/19.
//

#ifndef _TEST_FANOUT_FFNNET_ACTIVATION_H_
#define _TEST_FANOUT_FFNNET_ACTIVATION_H_

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
#include "ValarrayMap.h"

using flexnnet::ValarrayMap;

#define TESTCASE_PATH "test/nn_activation_tests/samples/"

struct FanoutTestCase
{
   size_t input1_sz;
   size_t input2_sz;

   size_t hlayer_sz;
   flexnnet::Array2D<double> hlayer_weights;

   size_t olayer1_sz;
   size_t olayer2_sz;

   flexnnet::Array2D<double> olayer1_weights;
   flexnnet::Array2D<double> olayer2_weights;

   flexnnet::ValarrayMap input;
   flexnnet::ValarrayMap target_output;
};

template<typename T>
class TestFanoutFFNNActivation : public ::testing::Test
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   std::string get_typeid();
   void create_fanout_ffnnet(const FanoutTestCase& _testcase);

   std::vector<FanoutTestCase> read_samples(std::string _fpath);
   ValarrayMap parse_datum(const rapidjson::Value& _obj);
   flexnnet::Array2D<double> parse_weights(const rapidjson::Value& _obj, size_t _rows, size_t _cols);

public:

   std::string HIDDEN_LAYER_TYPE_ID;
   std::string HIDDEN_LAYER_ID;

   std::string FANOUT_OLAYER1_ID;
   std::string FANOUT_OLAYER2_ID;

   std::set<std::string> LAYER_IDS;

   std::shared_ptr<flexnnet::BasicNeuralNet> nnet;
};

TYPED_TEST_CASE_P
(TestFanoutFFNNActivation);

typedef ::testing::Types<flexnnet::PureLin,
                         flexnnet::TanSig,
                         flexnnet::LogSig,
                         flexnnet::RadBas,
                         flexnnet::SoftMax> MyTypes;

template<typename T> std::string TestFanoutFFNNActivation<T>::get_typeid()
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

template<typename T> void TestFanoutFFNNActivation<T>::create_fanout_ffnnet(const FanoutTestCase& _testcase)
{
   // Set network basic_layer names
   HIDDEN_LAYER_TYPE_ID = TestFanoutFFNNActivation<T>::get_typeid();
   HIDDEN_LAYER_ID = HIDDEN_LAYER_TYPE_ID;
   std::transform(HIDDEN_LAYER_ID.begin(), HIDDEN_LAYER_ID.end(), HIDDEN_LAYER_ID.begin(), ::tolower);

   FANOUT_OLAYER1_ID = "output1";
   FANOUT_OLAYER2_ID = "output2";

   LAYER_IDS = {FANOUT_OLAYER1_ID, FANOUT_OLAYER2_ID, HIDDEN_LAYER_ID};

   std::vector<std::shared_ptr<flexnnet::OldNetworkLayer>> network_layers;

   // Create hidden basic_layer
   network_layers
      .push_back(std::shared_ptr<T>(new T(_testcase.hlayer_sz, HIDDEN_LAYER_ID, flexnnet::OldNetworkLayer::Hidden)));

   // Add external input to hidden basic_layer
   network_layers[0]->add_external_input(_testcase.input, {"input1"});

   // Set hidden basic_layer weights
   network_layers[0]->layer_weights.set(_testcase.hlayer_weights);

   // Create output basic_layer #1
   network_layers
      .push_back(std::shared_ptr<T>(new T(_testcase.olayer1_sz, FANOUT_OLAYER1_ID, flexnnet::OldNetworkLayer::Output)));
   network_layers[1]->add_connection(*network_layers[0], flexnnet::OldLayerConnRecord::Forward);

   // Set output basic_layer #2 weights
   network_layers[1]->layer_weights.set(_testcase.olayer1_weights);

   // Create output basic_layer #2
   network_layers
      .push_back(std::shared_ptr<T>(new T(_testcase.olayer2_sz, FANOUT_OLAYER2_ID, flexnnet::OldNetworkLayer::Output)));
   network_layers[2]->add_connection(*network_layers[0], flexnnet::OldLayerConnRecord::Forward);

   // Set output basic_layer #2 weights
   network_layers[2]->layer_weights.set(_testcase.olayer2_weights);

   // Create neural net
   nnet = std::shared_ptr<flexnnet::BasicNeuralNet>(new flexnnet::BasicNeuralNet(network_layers, false, "nnet"));
}

#endif //_TEST_FANOUT_FFNNET_ACTIVATION_H_
