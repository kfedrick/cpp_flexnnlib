//
// Created by kfedrick on 6/25/19.
//

#ifndef _TEST_SINGLE_LAYER_NNET_ACTIVATION_H_
#define _TEST_SINGLE_LAYER_NNET_ACTIVATION_H_

#include <gtest/gtest.h>

#include <string>

#include "BasicLayer.h"
#include "PureLin.h"
#include "TanSig.h"
#include "RadBas.h"
#include "SoftMax.h"
#include "LogSig.h"

using flexnnet::NNetIO_Typ;

#define TESTCASE_PATH "test/nn_activation_tests/samples/"

struct TestCase
{
   size_t layer_sz;
   size_t input_sz;
   flexnnet::Array2D<double> weights;

   NNetIO_Typ input;
   NNetIO_Typ target_output;
};

template<typename T>
class TestSingleLayerNNActivation : public ::testing::Test
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   std::string get_typeid();

   void create_single_layer_nnet(const TestCase& _testcase);
   void create_two_layer_ffnnet(const TestCase& _testcase);

   std::vector<TestCase> read_samples(std::string _fpath);
   NNetIO_Typ parse_datum(const rapidjson::Value& _obj);
   flexnnet::Array2D<double> parse_weights(const rapidjson::Value& _obj, size_t _rows, size_t _cols);

public:

   std::string SINGLE_LAYER_TYPE_ID;
   std::string SINGLE_LAYER_ID;

   std::set<std::string> LAYER_IDS;

   std::shared_ptr<flexnnet::BasicNeuralNet> nnet;

   std::string prettyPrintVector(const std::string& _label, const std::valarray<double>& _vec, int _prec=4)
   {
      std::stringstream ssout;
      ssout.precision(_prec);

      bool first = true;
      ssout << "\n\"" << _label << "\" : \n";
      ssout << "   [";
      for (auto& val : _vec)
      {
         if (!first)
            ssout << ", ";
         else
            first = false;

         ssout << val;
      }
      ssout << "]";

      return ssout.str();
   };
};

TYPED_TEST_CASE_P
(TestSingleLayerNNActivation);

typedef ::testing::Types<flexnnet::PureLin,
                         flexnnet::TanSig,
                         flexnnet::RadBas,
                         flexnnet::SoftMax,
                         flexnnet::LogSig> MyTypes;

template<typename T> std::string TestSingleLayerNNActivation<T>::get_typeid()
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

template<typename T> void TestSingleLayerNNActivation<T>::create_single_layer_nnet(const TestCase& _testcase)
{
   // Set network layer names
   SINGLE_LAYER_TYPE_ID = TestSingleLayerNNActivation<T>::get_typeid();
   SINGLE_LAYER_ID = SINGLE_LAYER_TYPE_ID;
   std::transform(SINGLE_LAYER_ID.begin(), SINGLE_LAYER_ID.end(), SINGLE_LAYER_ID.begin(), ::tolower);
   LAYER_IDS = {SINGLE_LAYER_ID};

   // Create layer
   std::vector<std::shared_ptr<flexnnet::NetworkLayer>> network_layers;
   network_layers
      .push_back(std::shared_ptr<T>(new T(_testcase.layer_sz, SINGLE_LAYER_ID, flexnnet::NetworkLayer::Output)));

   // Add external input to layer
   network_layers[0]->add_external_input(_testcase.input, {"input"});

   // Set layer weights
   network_layers[0]->layer_weights.set(_testcase.weights);

   // Create neural net
   nnet = std::shared_ptr<flexnnet::BasicNeuralNet>(new flexnnet::BasicNeuralNet(network_layers, false, "nnet"));
}

template<typename T> void TestSingleLayerNNActivation<T>::create_two_layer_ffnnet(const TestCase& _testcase)
{
   // Set network layer names
   SINGLE_LAYER_TYPE_ID = TestSingleLayerNNActivation<T>::get_typeid();
   SINGLE_LAYER_ID = SINGLE_LAYER_TYPE_ID;
   std::transform(SINGLE_LAYER_ID.begin(), SINGLE_LAYER_ID.end(), SINGLE_LAYER_ID.begin(), ::tolower);
   LAYER_IDS = {SINGLE_LAYER_ID, "output"};

   std::vector<std::shared_ptr<flexnnet::NetworkLayer>> network_layers;

   // Create hidden layer
   network_layers
      .push_back(std::shared_ptr<T>(new T(_testcase.layer_sz, SINGLE_LAYER_ID, flexnnet::NetworkLayer::Hidden)));

   // Add external input to layer
   network_layers[0]->add_external_input(_testcase.input, {"input"});

   // Set layer weights
   network_layers[0]->layer_weights.set(_testcase.weights);

   // Create output layer
   network_layers.push_back(std::shared_ptr<flexnnet::PureLin>(new flexnnet::PureLin(_testcase
                                                                                        .layer_sz, "output", flexnnet::NetworkLayer::Output)));

   // Add input from hidden layer
   network_layers[1]->add_connection(*network_layers[0], flexnnet::LayerConnRecord::Forward);

   // Create diagonal matrix for output layer weights
   flexnnet::Array2D<double> diag(_testcase.layer_sz, _testcase.layer_sz + 1);
   for (int i = 0; i < _testcase.layer_sz; i++)
      diag.at(i, i) = 1.0;
   network_layers[1]->layer_weights.set(diag);

   // Create neural net
   nnet = std::shared_ptr<flexnnet::BasicNeuralNet>(new flexnnet::BasicNeuralNet(network_layers, false, "nnet"));
}

#endif //_TEST_SINGLE_LAYER_NNET_ACTIVATION_H_
