//
// Created by kfedrick on 5/22/19.
//

#ifndef FLEX_NEURALNET_LAYERDERIVTESTCASE_H_
#define FLEX_NEURALNET_LAYERDERIVTESTCASE_H_

#include <string>
#include <vector>
#include <rapidjson/document.h>
#include "Array.h"

namespace flexnnet
{

   class LayerDerivTestCase
   {
   public:
      struct LayerDerivTestSample
      {
         std::vector<double> input;
         flexnnet::Array<double> target_dAdN;
         flexnnet::Array<double> target_dNdW;
         flexnnet::Array<double> target_dNdI;
      };

   public:
      LayerDerivTestCase ();
      virtual void read (const std::string &_filepath);

   protected:
      void readWeights ();
      void readTestCases ();

   public:
      std::string layer_name;
      std::string layer_type;
      unsigned int input_size;
      unsigned int layer_size;
      flexnnet::Array<double> weights;

      std::vector <LayerDerivTestSample> samples;

   private:
      rapidjson::Document doc;
   };
}

#endif //FLEX_NEURALNET_LAYERDERIVTESTCASE_H_
