//
// Created by kfedrick on 2/20/21.
//

#ifndef _NNBUILDERTESTFIXTURE_H_
#define _NNBUILDERTESTFIXTURE_H_

#include <fstream>

#include <CommonTestFixtureFunctions.h>

#include <flexnnet.h>
#include "PureLin.h"
#include "TanSig.h"
#include "LayerConnRecord.h"
#include "NetworkLayerImpl.h"
#include "NeuralNetTopology.h"

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>

using flexnnet::PureLin;
using flexnnet::TanSig;

using flexnnet::LayerConnRecord;
using flexnnet::NetworkLayerImpl;
using flexnnet::NeuralNetTopology;

class NNBuilderTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   struct TestCaseRecord
   {
      std::valarray<double> initial_value;
      std::valarray<double> input;
      flexnnet::Array2D<double> weights;

      struct Target
      {
         std::valarray<double> output;
         flexnnet::Array2D<double> dAdN;
         flexnnet::Array2D<double> dNdW;
         flexnnet::Array2D<double> dNdI;
      };

      Target target;
   };

   NNBuilderTestFixture()
   {
      sample_external_input = {{"a", {1, 2}}, {"b", {0.1, -2, 1.5}}, {"c", {1e10, 2.5, 666, 7}}};
   }

   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   flexnnet::ValarrMap sample_external_input;

public:
   void readTestCases(const std::string& _filepath);
   std::vector<TestCaseRecord> samples;

private:
   rapidjson::Document doc;
};

void NNBuilderTestFixture::readTestCases(const std::string& _filepath)
{
   samples.clear();

   // Open file and create rabidjson file stream wrapper
   std::cout << _filepath << "\n";
   std::ifstream in_fstream(_filepath);
   rapidjson::IStreamWrapper in_fswrapper(in_fstream);

   // Parse json file stream into rapidjson document
   doc.ParseStream(in_fswrapper);

   //Save basic_layer identifier information
   rapidjson::Value layer_def_obj = doc["layer_definition"].GetObject();

   int layer_sz = layer_def_obj["dimensions"]["layer_size"].GetUint();
   int layer_input_sz = layer_def_obj["dimensions"]["layer_input_size"].GetUint();

   // Read layer weights and set them
   const rapidjson::Value& weights_arr = layer_def_obj["learned_parameters"]["weights"];
   flexnnet::Array2D<double> weights;

   weights.resize(layer_sz, layer_input_sz + 1);
   for (rapidjson::SizeType i = 0; i < weights_arr.Size(); i++)
   {
      const rapidjson::Value& myrow = weights_arr[i];
      for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
         weights.at(i, j) = myrow[j].GetDouble();
   }

   const rapidjson::Value& test_cases_arr = doc["test_cases"].GetArray();

   /*
    * Iterate through all of the test pairs in the "test_pairs" vector
    */
   for (rapidjson::SizeType i = 0; i < test_cases_arr.Size(); i++)
   {
      // save a reference to the i'th test pair
      const rapidjson::Value& a_tuple_obj = test_cases_arr[i];

      // Set the input vector to the correct size and copy the sample input vector
      TestCaseRecord test_sample;

      test_sample.input.resize(layer_input_sz);
      test_sample.initial_value.resize(layer_sz);
      test_sample.target.output.resize(layer_sz);
      test_sample.weights.set(weights);

      // Read the test input vector
      const rapidjson::Value& in_arr = a_tuple_obj["input"];
      for (rapidjson::SizeType i = 0; i < in_arr.Size(); i++)
         test_sample.input[i] = in_arr[i].GetDouble();

      // Read the test input vector
      const rapidjson::Value& init_arr = a_tuple_obj["initial_value"];
      for (rapidjson::SizeType i = 0; i < init_arr.Size(); i++)
         test_sample.initial_value[i] = init_arr[i].GetDouble();

      // Read the test input vector
      const rapidjson::Value& out_arr = a_tuple_obj["target"]["output"];
      for (rapidjson::SizeType i = 0; i < out_arr.Size(); i++)
         test_sample.target.output[i] = out_arr[i].GetDouble();

      // Read dy_dnet
      const rapidjson::Value& dAdN_arr = a_tuple_obj["target"]["dAdN"];
      test_sample.target.dAdN.resize(layer_sz, layer_sz);
      for (rapidjson::SizeType i = 0; i < dAdN_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dAdN_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
            test_sample.target.dAdN.at(i, j) = myrow[j].GetDouble();
      }

      // Read dnet_dw
      const rapidjson::Value& dNdW_arr = a_tuple_obj["target"]["dNdW"];
      test_sample.target.dNdW.resize(layer_sz, layer_input_sz + 1);
      for (rapidjson::SizeType i = 0; i < dNdW_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dNdW_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
            test_sample.target.dNdW.at(i, j) = myrow[j].GetDouble();
      }

      // Read dnet_dx
      const rapidjson::Value& dNdI_arr = a_tuple_obj["target"]["dNdI"];
      test_sample.target.dNdI.resize(layer_sz, layer_input_sz);
      for (rapidjson::SizeType i = 0; i < dNdI_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dNdI_arr[i];
         for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
            test_sample.target.dNdI.at(i, j) = myrow[j].GetDouble();
      }

      // Push a copy of the test pair onto the test pairs vector
      samples.push_back(test_sample);
   }
}


TEST_F(NNBuilderTestFixture, PureLinNetworkLayer)
{
   try
   {
      NetworkLayerImpl<PureLin> nl(3, "purelin");
      ASSERT_EQ(nl.size(), 3)
         << "Expected network layer size to be 3, got " << nl.size() << "\n";
      ASSERT_EQ(nl.input_size(), 0)
         << "Expected network layer input size to be 0, got " << nl.input_size() << "\n";
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}


TEST_F(NNBuilderTestFixture, NetworkLayerAddExtInput)
{
   try
   {
      NetworkLayerImpl<PureLin> nl(3, "purelin");
      nl.add_external_input_field("input1", 1);

      ASSERT_EQ(nl.input_size(), 1) << "Expected external fields size to be zero, got " << nl.input_size() << "\n";
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NetworkLayerAddLayerConn)
{
   try
   {
      auto ol_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(1, "output"));
      auto hl_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(3, "hidden"));

      ol_ptr->add_connection("activation", hl_ptr, LayerConnRecord::Forward);

      ASSERT_EQ(ol_ptr->input_size(), 3)
         << "Expected input size to be 3, got " << ol_ptr->input_size() << "\n";
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NetworkLayerDupLayerConn)
{

      auto ol_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(1, "output"));
      auto hl_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(3, "hidden"));

      ol_ptr->add_connection("activation", hl_ptr, LayerConnRecord::Forward);
      EXPECT_THROW(ol_ptr->add_connection("activation", hl_ptr, LayerConnRecord::Forward), std::invalid_argument);
}

TEST_F(NNBuilderTestFixture, NetworkLayerFaninLayerConn)
{
   try
   {
      auto ol_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(1, "output"));
      auto hl1_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(2, "hidden1"));
      auto hl2_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(3, "hidden2"));

      ol_ptr->add_connection("activation", hl1_ptr, LayerConnRecord::Forward);
      ol_ptr->add_connection("activation", hl2_ptr, LayerConnRecord::Forward);

      ASSERT_EQ(ol_ptr->input_size(), 5)
                     << "Expected input size to be 5, got " << ol_ptr->input_size() << "\n";
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NNTopoEmptyDecl)
{
   try
   {
      NeuralNetTopology topo();
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NNTopoAddSingleLayer)
{
   NeuralNetTopology topo;
   auto nl_ptr = std::make_shared<NetworkLayerImpl<PureLin>>(NetworkLayerImpl<PureLin>(3, "purelin"));

   try
   {
      topo.network_layers[nl_ptr->name()] = nl_ptr;
      ASSERT_EQ(topo.network_layers.size(), 1)
         << "Expected 1 laye in NeuralNetTopology::network_layers, got " << topo.network_layers.size() << "\n";
   }
   catch (std::exception& err)
   {
      FAIL() << "Couldn't find layer \"output\" we just added.\n" << err.what();
   }
}

TEST_F(NNBuilderTestFixture, TanSigActivation)
{
   printf("TEST TanSig Network Layer activate\n");

   readTestCases("tansig_activity_test1.json");

   flexnnet::ValarrMap inputmap;
   std::valarray<double> errv;
   flexnnet::ValarrMap errvmap;
   for (auto& item : samples)
   {
      size_t layer_sz = item.target.output.size();
      size_t layer_in_sz = item.input.size();

      errv.resize(layer_sz);
      errv = 1;
      //errv[0] = 1;
      errvmap["output"] = errv;

      auto l_ptr = std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(layer_sz, "output", TanSig::DEFAULT_PARAMS, true));
      l_ptr->add_external_input_field("input", layer_in_sz);

      auto dim = item.weights.size();
      std::cout << dim.rows << " " << dim.cols << "\n";

      l_ptr->set_weights(item.weights);
      std::cout << prettyPrintArray("layer weights", item.weights);

      inputmap["input"] = item.input;
      const std::valarray<double>& outv = l_ptr->activate(inputmap);

      // Check basic_layer output
      EXPECT_PRED3(vector_double_near, item.target.output, outv, 0.000000001) << "ruh roh";

      std::cout << prettyPrintVector("inputv", item.input);
      std::cout << prettyPrintVector("layer activate outputv", outv);

      l_ptr->backprop(errvmap);

      auto dEdw = l_ptr->dEdw();
      std::cout << prettyPrintArray("dEdw", dEdw, 15);

   }
}


#endif //_NNBUILDERTESTFIXTURE_H_
