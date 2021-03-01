//
// Created by kfedrick on 6/19/19.
//

#include "test_single_layer_nnet.h"

#include <map>
#include <valarray>

#include "Datum.h"
#include "BasicNeuralNetFactory.h"

using std::map;
using std::valarray;
using flexnnet::Array2D;
using flexnnet::Datum;
using flexnnet::OldNetworkLayer;
using flexnnet::BasicNeuralNet;
using flexnnet::BasicNeuralNetFactory;

TYPED_TEST_P (TestSingleLayerNNBuild, SingleLayerNNBuild)
{
   std::string transfunc_type_str = typeid(TypeParam).name();
   static char buf[2048];
   size_t size = sizeof(buf);
   int status;

   char* res = abi::__cxa_demangle(transfunc_type_str.c_str(), buf, &size, &status);
   buf[sizeof(buf) - 1] = 0;

   const string LAYER_ID = buf;
   const size_t LAYER_SZ = 3;
   const size_t INPUT_SZ = 5;
   const BasicLayer::NetworkLayerType LAYER_LTYPE = BasicLayer::Output;
   const bool IS_OUTPUT_LAYER = (LAYER_LTYPE == BasicLayer::Output);

   // Set network input datum sample
   std::valarray<double> invec(INPUT_SZ);
   Datum DATUM({{"input", invec}});

   // Set target network basic_layer names
   std::set<string> LAYER_NAMES = {LAYER_ID};

   std::cout << "\n Single Layer (" << LAYER_ID.c_str() << ") NN Basic Build Test\n";

   // Set target single network basic_layer weights size
   Array2D<double>::Dimensions WEIGHTS_SZ = {.rows=LAYER_SZ, .cols=DATUM[0].size() + 1};
   Array2D<double>::Dimensions DADN_SZ = {.rows=LAYER_SZ, .cols=LAYER_SZ};
   Array2D<double>::Dimensions DNDW_SZ = {.rows=LAYER_SZ, .cols=DATUM[0].size() + 1};
   Array2D<double>::Dimensions DNDI_SZ = {.rows=LAYER_SZ, .cols=DATUM[0].size() + 1};

   // Construct and build network
   BasicNeuralNetFactory factory;

   std::shared_ptr<TypeParam> layer_ptr = factory
      .add_layer<TypeParam>(LAYER_SZ, LAYER_ID, LAYER_LTYPE);

   factory.set_layer_external_input(LAYER_ID, DATUM, {"input"});

   std::shared_ptr<BasicNeuralNet> nnet = factory.build("test_nn");

   // Verify that network basic_layer names are identical
   std::set<string> layer_names = nnet->get_layer_names();
   ASSERT_EQ(LAYER_NAMES, layer_names);

   // Get list of network layers
   const std::vector<std::shared_ptr<OldNetworkLayer>>& layers = nnet->get_layers();

   // There should be exactly one basic_layer
   ASSERT_EQ(1, layers.size());

   // Verify basic basic_layer info: id, size, input size, is_output
   const BasicLayer& layer_ref = *layers[0];
   ASSERT_EQ(LAYER_ID, layer_ref.name());
   ASSERT_EQ(LAYER_SZ, layer_ref.size());
   ASSERT_EQ(DATUM["input"].size(), layer_ref.input_size());
   ASSERT_EQ(IS_OUTPUT_LAYER, layer_ref.is_output_layer());

   // Verify weight array size
   const Array2D<double>& weights = layer_ref.layer_weights.const_weights_ref;
   ASSERT_EQ(WEIGHTS_SZ.rows, weights.size().rows);
   ASSERT_EQ(WEIGHTS_SZ.cols, weights.size().cols);

   // Verify partial derivative array size (d_output wrt net input)
   const Array2D<double>& dAdN = layer_ref.get_dAdN();
   ASSERT_EQ(DADN_SZ.rows, dAdN.size().rows);
   ASSERT_EQ(DADN_SZ.cols, dAdN.size().cols);

   // Verify partial derivative array size (d_netin wrt weights)
   const Array2D<double>& dNdW = layer_ref.get_dNdW();
   ASSERT_EQ(DNDW_SZ.rows, dNdW.size().rows);
   ASSERT_EQ(DNDW_SZ.cols, dNdW.size().cols);

   // Verify partial derivative array size (d_netin wrt raw input)
   const Array2D<double>& dNdI = layer_ref.get_dNdI();
   ASSERT_EQ(DNDI_SZ.rows, dNdI.size().rows);
   ASSERT_EQ(DNDI_SZ.cols, dNdI.size().cols);
}

REGISTER_TYPED_TEST_CASE_P
(TestSingleLayerNNBuild, SingleLayerNNBuild);
INSTANTIATE_TYPED_TEST_CASE_P
(My, TestSingleLayerNNBuild, MyTypes);