//
// Created by kfedrick on 5/23/19.
//

#include "PureLinDerivTestCase.h"
#include <iostream>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

void flexnnet::PureLinDerivTestCase::read (const std::string &_filepath)
{
   LayerDerivTestCase::read(_filepath);

   // Open file and create rabidjson file stream wrapper
   std::ifstream in_fstream (_filepath);
   rapidjson::IStreamWrapper in_fswrapper (in_fstream);

   // Parse json file stream into rapidjson document
   rapidjson::Document doc;
   doc.ParseStream (in_fswrapper);

   gain = doc["transfer_function_params"]["gain"].GetDouble();

   in_fstream.close();

   printf ("\n\ngain %f\n", gain);
}