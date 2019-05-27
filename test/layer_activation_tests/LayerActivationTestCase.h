//
// Created by kfedrick on 5/19/19.
//

#ifndef FLEX_NEURALNET_LAYERACTIVATIONTESTCASE_H_
#define FLEX_NEURALNET_LAYERACTIVATIONTESTCASE_H_

#include <string>
#include <vector>
#include <rapidjson/document.h>
#include "Array.h"

namespace flexnnet
{
   class LayerActivationTestCase
   {
   public:
      struct LayerActivationTestPair
      {
         std::vector<double> input;
         std::vector<double> target;
      };

   public:
      LayerActivationTestCase ();
      virtual void read (const std::string &_filepath);

   protected:
      void readWeights ();
      void readTestPairs();

   public:
      std::string layer_name;
      std::string layer_type;
      unsigned int input_size;
      unsigned int layer_size;
      flexnnet::Array<double> weights;

      std::vector<LayerActivationTestPair> samples;

   private:
      rapidjson::Document doc;
   };
}

#endif //FLEX_NEURALNET_LAYERACTIVATIONTESTCASE_H_
