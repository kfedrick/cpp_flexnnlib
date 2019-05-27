//
// Created by kfedrick on 5/19/19.
//

#include "LayerActivationTestCase.h"

#include <iostream>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

flexnnet::LayerActivationTestCase::LayerActivationTestCase()
{

}


void flexnnet::LayerActivationTestCase::read(const std::string& _filepath)
{
   // Open file and create rabidjson file stream wrapper
   std::ifstream in_fstream (_filepath);
   rapidjson::IStreamWrapper in_fswrapper (in_fstream);

   // Parse json file stream into rapidjson document
   doc.ParseStream (in_fswrapper);

   //Save layer identifier information
   layer_name = doc["id"]["name"].GetString();
   layer_type = doc["id"]["type"].GetString();

   //Save layer dimension information
   layer_size = doc["dimensions"]["layer_size"].GetUint();
   input_size = doc["dimensions"]["layer_input_size"].GetUint();

   //Save layer weights information - resize weights as required
   weights.resize(layer_size, input_size+1);
   readWeights ();

   // Read input/output test pairs
   readTestPairs();

   in_fstream.close();
}

/**
 * Read layer weights from rapidjson::Document and store them in the weights
 * data member
 */
void flexnnet::LayerActivationTestCase::readWeights ()
{
   const rapidjson::Value &weights_obj = doc["learning_params"]["weights"].GetArray();

   for (rapidjson::SizeType i=0; i<weights_obj.Size(); i++)
   {
      const rapidjson::Value& myrow = weights_obj[i];

      for (rapidjson::SizeType j=0; j<myrow.Size(); j++)
         weights[i][j] = myrow[j].GetDouble();
   }
}

/**
 * Read input/output test pairs from rapidjson::Document and store them in the
 * test_pairs data data member
 */
void flexnnet::LayerActivationTestCase::readTestPairs()
{
   const rapidjson::Value& test_pairs_arr = doc["test_pairs"].GetArray();

   /*
    * Iterate through all of the test pairs in the "test_pairs" vector
    */
   for (rapidjson::SizeType i=0; i<test_pairs_arr.Size(); i++)
   {
      // save a reference to the i'th test pair
      const rapidjson::Value& a_pair_obj = test_pairs_arr[i];

      // Set the input and target vectors to the correct sizes
      LayerActivationTestPair test_pair;
      test_pair.input.resize(input_size);
      test_pair.target.resize(layer_size);

      // Read the test input vector
      const rapidjson::Value &in_arr = a_pair_obj["input"];
      for (rapidjson::SizeType i = 0; i < in_arr.Size (); i++)
         test_pair.input[i] = in_arr[i].GetDouble ();

      // Read and save the test target output vector
      const rapidjson::Value& out_arr = a_pair_obj["target"];
      for (rapidjson::SizeType i = 0; i < out_arr.Size (); i++)
         test_pair.target[i] = out_arr[i].GetDouble ();

      // Push a copy of the test pair onto the test pairs vector
      samples.push_back(test_pair);
   }
}
