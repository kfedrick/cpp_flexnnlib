//
// Created by kfedrick on 5/22/19.
//

#include "LayerDerivTestCase.h"

#include <iostream>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

flexnnet::LayerDerivTestCase::LayerDerivTestCase()
{

}


void flexnnet::LayerDerivTestCase::read(const std::string& _filepath)
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
   readTestCases ();

   in_fstream.close();
}

/**
 * Read layer weights from rapidjson::Document and store them in the weights
 * data member
 */
void flexnnet::LayerDerivTestCase::readWeights ()
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
void flexnnet::LayerDerivTestCase::readTestCases ()
{
   const rapidjson::Value& test_cases_arr = doc["test_cases"].GetArray();

   /*
    * Iterate through all of the test pairs in the "test_pairs" vector
    */
   for (rapidjson::SizeType i=0; i<test_cases_arr.Size(); i++)
   {
      // save a reference to the i'th test pair
      const rapidjson::Value& a_pair_obj = test_cases_arr[i];

      // Set the input and target vectors to the correct sizes
      LayerDerivTestSample test_pair;
      test_pair.input.resize(input_size);

      // Read the test input vector
      const rapidjson::Value &in_arr = a_pair_obj["input"];
      for (rapidjson::SizeType i = 0; i < in_arr.Size (); i++)
         test_pair.input[i] = in_arr[i].GetDouble ();

      // Read dAdN
      const rapidjson::Value& dAdN_arr = a_pair_obj["dAdN"];
      test_pair.target_dAdN.resize(layer_size, layer_size);
      for (rapidjson::SizeType i=0; i<dAdN_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dAdN_arr[i];
         for (rapidjson::SizeType j=0; j<myrow.Size(); j++)
            test_pair.target_dAdN[i][j] = myrow[j].GetDouble();
      }

      // Read dNdW
      const rapidjson::Value& dNdW_arr = a_pair_obj["dNdW"];
      test_pair.target_dNdW.resize(layer_size, input_size+1);
      for (rapidjson::SizeType i=0; i<dNdW_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dNdW_arr[i];
         for (rapidjson::SizeType j=0; j<myrow.Size(); j++)
            test_pair.target_dNdW[i][j] = myrow[j].GetDouble();
      }

      // Read dNdI
      const rapidjson::Value& dNdI_arr = a_pair_obj["dNdI"];
      test_pair.target_dNdI.resize(layer_size, input_size+1);
      for (rapidjson::SizeType i=0; i<dNdI_arr.Size(); i++)
      {
         const rapidjson::Value& myrow = dNdI_arr[i];
         for (rapidjson::SizeType j=0; j<myrow.Size(); j++)
            test_pair.target_dNdI[i][j] = myrow[j].GetDouble();
      }

      // Push a copy of the test pair onto the test pairs vector
      samples.push_back(test_pair);
   }
}
