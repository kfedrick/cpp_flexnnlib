//
// Created by kfedrick on 6/17/19.
//

#ifndef _TEST_LOGSIG_CONSTRUCTOR_H_
#define _TEST_LOGSIG_CONSTRUCTOR_H_

#include <gtest/gtest.h>
#include "test_layer_constructor.h"

#include "LogSig.h"

using std::string;

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::LogSig;


TEST_F(TestLayerConstructors, ConstructLogSigWithDefaults)
{
   const int OUT_SZ = 2;
   const string NAME = "test1";
   BasicLayer::NetworkLayerType network_layer_type = BasicLayer::Output;

   // Create basic_layer
   LogSig layer(OUT_SZ, NAME, network_layer_type);

   // Name and size should be as set_weights in constructor
   ASSERT_EQ(NAME, layer.name());
   ASSERT_EQ(OUT_SZ, layer.size());

   // Layer type should be Output by default
   ASSERT_EQ(true, layer.is_output_layer());

   // Initial input size is zero
   ASSERT_EQ(0, layer.virtual_input_size());

   // gain is default value
   ASSERT_EQ(LogSig::DEFAULT_PARAMS.gain, layer.get_gain());
}

TEST_F(TestLayerConstructors, ConstructLogSigWithInputSize)
{
   const int OUT_SZ = 2;
   const int RAWIN_SZ = 5;
   const string NAME = "test2";
   BasicLayer::NetworkLayerType network_layer_type = BasicLayer::Output;

   // Create basic_layer
   LogSig layer(OUT_SZ, NAME, network_layer_type);
   layer.resize_input(RAWIN_SZ);

   // id, size, and input sizes should be as set_weights.
   ASSERT_EQ(NAME, layer.name());
   ASSERT_EQ(OUT_SZ, layer.size());
   ASSERT_EQ(RAWIN_SZ, layer.input_size());

   // Layer type should be Output by default
   ASSERT_EQ(true, layer.is_output_layer());

   // Weight array should be output size rows by input_sz+1 columns
   Array2D<double>::Dimensions dim = layer.layer_weights.const_weights_ref.size();
   ASSERT_EQ(OUT_SZ, dim.rows);
   ASSERT_EQ(RAWIN_SZ + 1, dim.cols);
}

TEST_F(TestLayerConstructors, ConstructLogSigWithLayerTypeHidden)
{
   const int OUT_SZ = 2;
   const string NAME = "test2";
   BasicLayer::NetworkLayerType network_layer_type = BasicLayer::Hidden;

   // Create basic_layer
   LogSig layer(OUT_SZ, NAME, network_layer_type);

   // Layer type should be Hidden by default
   ASSERT_EQ(false, layer.is_output_layer());
}

TEST_F(TestLayerConstructors, ConstructLogSigWithParams)
{
   const int OUT_SZ = 2;
   const string NAME = "test2";
   BasicLayer::NetworkLayerType network_layer_type = BasicLayer::Hidden;
   LogSig::Parameters PARAMS = {.gain = 1.27};

   // Create basic_layer
   LogSig layer(OUT_SZ, NAME, network_layer_type, PARAMS);

   // Layer type should be Hidden by default
   ASSERT_EQ(PARAMS.gain, layer.get_gain());
}

#endif //_TEST_LOGSIG_CONSTRUCTOR_H_
