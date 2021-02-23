//
// Created by kfedrick on 6/17/19.
//

#include "test_layer_serialization.h"

#include "RadBas.h"

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::RadBas;

/**
 * Test serialization of 2x5 RadBas basiclayer
 */
TEST_F(TestLayerSerialization, SerializeRadBas)
{
   const std::string target_json = "{\"id\":\"test\",\"is_output_layer\":true,\"dimensions\":{\"layer_size\":2,\"layer_input_size\":5},\"learned_parameters\":{\"weights\":[[1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,-1.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::RadBas\",\"parameters\":{\"rescaled\":false}}}";

   const int OUT_SZ = 2;
   const int IN_SZ = 5;
   const std::string NAME = "test";

   // Construct basiclayer
   RadBas layer(OUT_SZ, NAME, BasicLayer::Output);
   layer.resize_input(IN_SZ);

   // Set weights
   flexnnet::Array2D<double> initw(OUT_SZ, IN_SZ + 1);
   initw = 0;
   initw.at(0, 0) = 1.0;
   initw.at(1, 2) = 2.0;
   initw.at(1, 2) = -1.0;
   layer.layer_weights.set(initw);

   // Set gain
   layer.set_rescaled(false);

   // Get serialized basiclayer as Json string
   std::string output_json = layer.toJson();
   std::cout << "\n\n" << output_json << std::endl;

   ASSERT_EQ(target_json, output_json);
}

TEST_F(TestLayerSerialization, DeserializeRadBas)
{
   const std::string NAME = "test";
   const unsigned int OUT_SZ = 2;
   const unsigned int RAWIN_SZ = 5;
   const bool RESCALED = false;

   const std::string json = "{\"id\":\"test\",\"is_output_layer\":true,\"dimensions\":{\"layer_size\":2,\"layer_input_size\":5},\"learned_parameters\":{\"weights\":[[1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,-1.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::RadBas\",\"parameters\":{\"rescaled\":false}}}";

   std::shared_ptr<RadBas> layer_ptr = flexnnet::LayerSerializer<RadBas>::parse(json);
   RadBas& layer = (*layer_ptr);

   // Check that constructor arguments are correct
   ASSERT_EQ(OUT_SZ, layer.size());
   ASSERT_EQ(RAWIN_SZ, layer.input_size());
   ASSERT_EQ(NAME, layer.name());
   ASSERT_EQ(RESCALED, layer.is_rescaled());

   // Set weights
   flexnnet::Array2D<double> target_weights(OUT_SZ, RAWIN_SZ + 1);
   target_weights = 0;
   target_weights.at(0, 0) = 1.0;
   target_weights.at(1, 2) = 2.0;
   target_weights.at(1, 2) = -1.0;

   // Check weights
   const Array2D<double>& weights = layer.layer_weights.const_weights_ref;
   EXPECT_PRED3(array_double_near, target_weights, weights, 0.000000001) << this->printArray("weights", weights);
}