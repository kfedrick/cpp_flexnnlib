//
// Created by kfedrick on 6/25/19.
//

#ifndef _TEST_SINGLE_LAYER_NNET_TRAINING_H_
#define _TEST_SINGLE_LAYER_NNET_TRAINING_H_

#include <gtest/gtest.h>

#include <string>

#include "BasicLayer.h"
#include "PureLin.h"
#include "TanSig.h"
#include "RadBas.h"
#include "SoftMax.h"
#include "LogSig.h"

#include "Exemplar.h"
#include "NeuralNet.h"

//#include "NNetTrainer.h"
//#include "TrainingAlgo.h"
#include "FATrainer.h"
#include "FA2Trainer.h"
#include "FuncApproxEvaluator.h"

#define TESTCASE_PATH "test/nn_basic_training_tests/samples/"

struct TestCase
{
   size_t layer_sz;
   size_t input_sz;
   flexnnet::Array2D<double> weights;

   flexnnet::Datum input;
   flexnnet::Datum target_output;
};

template<typename T>
class TestSingleLayerNNTraining : public ::testing::Test
{
public:
   virtual void SetUp ()
   {}
   virtual void TearDown ()
   {}

   std::string get_typeid ();

   void train_single_layer_nnet (const TestCase &_testcase);
   void create_two_layer_ffnnet (const TestCase& _testcase);

   std::vector<TestCase> read_samples(std::string _fpath);
   flexnnet::Datum parse_datum (const rapidjson::Value &_obj);
   flexnnet::Array2D<double> parse_weights (const rapidjson::Value &_obj, size_t _rows, size_t _cols);

public:

   std::string SINGLE_LAYER_TYPE_ID;
   std::string SINGLE_LAYER_ID;

   std::set<std::string> LAYER_IDS;

   std::vector<std::shared_ptr<flexnnet::NetworkLayer>> network_layers;

   std::shared_ptr<flexnnet::NeuralNet<flexnnet::Datum,flexnnet::Datum>> nnet;

   //std::shared_ptr<flexnnet::NNetTrainer<flexnnet::Datum, flexnnet::Datum, flexnnet::NeuralNet, flexnnet::Exemplar, flexnnet::TrainingAlgo>> basic_trainer;
   std::shared_ptr<flexnnet::FA2Trainer> basic_trainer;
};

TYPED_TEST_CASE_P (TestSingleLayerNNTraining);

typedef ::testing::Types<flexnnet::PureLin, flexnnet::TanSig, flexnnet::RadBas, flexnnet::SoftMax, flexnnet::LogSig> MyTypes;

template<typename T> std::string TestSingleLayerNNTraining<T>::get_typeid ()
{
   std::string type_id = typeid (T).name();
   static char buf[2048];
   size_t size = sizeof (buf);
   int status;

   char *res = abi::__cxa_demangle (type_id.c_str (), buf, &size, &status);
   buf[sizeof (buf) - 1] = 0;

   std::string buf_str = buf;
   size_t pos = buf_str.rfind (':')+1;
   buf_str = buf_str.substr(pos);

   return buf_str;
}

template<typename T> void TestSingleLayerNNTraining<T>::train_single_layer_nnet (const TestCase &_testcase)
{
   // Set network layer names
   SINGLE_LAYER_TYPE_ID = TestSingleLayerNNTraining<T>::get_typeid();
   SINGLE_LAYER_ID = SINGLE_LAYER_TYPE_ID;
   std::transform(SINGLE_LAYER_ID.begin(), SINGLE_LAYER_ID.end(), SINGLE_LAYER_ID.begin(), ::tolower);
   LAYER_IDS = {SINGLE_LAYER_ID};

   // Create layer
   network_layers.clear();
   network_layers.push_back(std::shared_ptr<T>(new T(_testcase.layer_sz, SINGLE_LAYER_ID, flexnnet::BasicLayer::Output)));

   // Add external input to layer
   network_layers[0]->add_external_input (_testcase.input, {"input"});

   // Set layer weights
   network_layers[0]->layer_weights.set_weights(_testcase.weights);

   // Create neural net
   nnet = std::shared_ptr<flexnnet::NeuralNet<flexnnet::Datum,flexnnet::Datum>>(new flexnnet::NeuralNet<flexnnet::Datum,flexnnet::Datum>(network_layers, false, SINGLE_LAYER_TYPE_ID));

   //std::shared_ptr<flexnnet::TrainingAlgo<flexnnet::Datum, flexnnet::Datum, flexnnet::Exemplar>> basic_train_algo = std::shared_ptr<flexnnet::TrainingAlgo<flexnnet::Datum, flexnnet::Datum, flexnnet::Exemplar>>(new flexnnet::TrainingAlgo<flexnnet::Datum, flexnnet::Datum, flexnnet::Exemplar>());
   //basic_trainer = std::shared_ptr<flexnnet::NNetTrainer<flexnnet::Datum, flexnnet::Datum, flexnnet::NeuralNet, flexnnet::Exemplar, flexnnet::TrainingAlgo>>(new  flexnnet::NNetTrainer<flexnnet::Datum, flexnnet::Datum, flexnnet::NeuralNet, flexnnet::Exemplar, flexnnet::TrainingAlgo>(*nnet, *basic_train_algo));
   basic_trainer = std::shared_ptr<flexnnet::FA2Trainer>(new  flexnnet::FA2Trainer());

}



#endif //_TEST_SINGLE_LAYER_NNET_TRAINING_H_
