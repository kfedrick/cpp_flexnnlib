//
// Created by kfedrick on 5/16/19.
//

#ifndef FLEX_NEURALNET_JSONENCODER_H_
#define FLEX_NEURALNET_JSONENCODER_H_

#include <document.h>
#include <stringbuffer.h>
#include <writer.h>
#include <prettywriter.h>

#include <string>
#include <vector>
#include <Array2D.h>

namespace flexnnet
{

   class JSONEncoder
   {
   protected:
      static void vectorToJSONObj (rapidjson::Value &_val, const std::vector<double> &_vec);
      static void ArrayToJSONObj (rapidjson::Value &_val, const flexnnet::Array2D<double> &_arr);
      static void JSONObjToArray (flexnnet::Array2D<double> &_arr, const rapidjson::Value &_val);

   protected:
      // Document to use for constructing this document
      static rapidjson::Document document;

      // Document::AllocatorType to use for constructing this document
      static rapidjson::Document::AllocatorType &allocator;
   };


}

#endif //FLEX_NEURALNET_JSONENCODER_H_
