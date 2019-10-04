//
// Created by kfedrick on 6/17/19.
//

#include "test_layer_serialization.h"

#include "LogSig.h"

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::LogSig;

/**
 * Test serialization of 2x5 LogSig layer
 */
TEST_F(TestLayerSerialization, SerializeLogSig)
{
   const std::string target_json = "{\"id\":\"test\",\"is_output_layer\":true,\"dimensions\":{\"layer_size\":2,\"layer_input_size\":5},\"learned_parameters\":{\"weights\":[[1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,-1.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::LogSig\",\"parameters\":{\"gain\":0.33}}}";
   const int OUT_SZ = 2;
   const int IN_SZ = 5;
   const std::string NAME = "test";

   // Construct layer
   LogSig layer (OUT_SZ, NAME, BasicLayer::Output);
   layer.resize_input (IN_SZ);

   // Set weights
   flexnnet::Array2D<double> initw (OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw.at (0, 0) = 1.0;
   initw.at (1, 2) = 2.0;
   initw.at (1, 2) = -1.0;
   layer.layer_weights.set_weights (initw);

   // Set gain
   layer.set_gain (0.33);

   // Get serialized layer as Json string
   std::string output_json = layer.toJson ();
   std::cout << "\n\n" << output_json << std::endl;

   ASSERT_EQ(target_json, output_json);
}

TEST_F(TestLayerSerialization, DeserializeLogSig)
{
   const std::string NAME = "test";
   const unsigned int OUT_SZ = 2;
   const unsigned int RAWIN_SZ = 5;
   const double GAIN = 0.666;

   std::string json = "{\"id\":\"test\",\"is_output_layer\":true,\"dimensions\":{\"layer_size\":2,\"layer_input_size\":5},\"learned_parameters\":{\"weights\":[[1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,-1.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::LogSig\",\"parameters\":{\"gain\":0.666}}}";

   std::shared_ptr<LogSig> layer_ptr = flexnnet::LayerSerializer<LogSig>::parse (json);
   LogSig &layer = (*layer_ptr);

   // Check that constructor arguments are correct
   ASSERT_EQ(OUT_SZ, layer.size ());
   ASSERT_EQ(RAWIN_SZ, layer.input_size ());
   ASSERT_EQ(NAME, layer.name ());
   ASSERT_EQ(GAIN, layer.get_gain ());

   // Set weights
   flexnnet::Array2D<double> target_weights (OUT_SZ, RAWIN_SZ + 1);
   target_weights = 0;
   target_weights.at (0, 0) = 1.0;
   target_weights.at (1, 2) = 2.0;
   target_weights.at (1, 2) = -1.0;

   // Check weights
   const Array2D<double>& weights = layer.layer_weights.const_weights_ref;
   EXPECT_PRED3(array_double_near, target_weights, weights, 0.000000001) << this->printArray("weights", weights);
}