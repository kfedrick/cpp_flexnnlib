/*
 * Parameters.cpp
 *
 *  Created on: Feb 2, 2014
 *      Author: kfedrick
 */

#include <stdexcept>
#include <sstream>

#include "Parameters.h"

using namespace std;

namespace flex_neuralnet
{

Parameters::Parameters(const string& id)
{
   class_id = id;
}

Parameters::~Parameters()
{
}

void Parameters::setparam(const string& key, bool val)
{
   throw invalid_argument(error(key, "bool").str());
}

void Parameters::setparam(const string& key, int val)
{
   throw invalid_argument(error(key, "int").str());
}

void Parameters::setparam(const string& key, float val)
{
   throw invalid_argument(error(key, "float").str());
}

void Parameters::setparam(const string& key, double val)
{
   throw invalid_argument(error(key, "double").str());
}

void Parameters::setparam(const string& key, const char* val)
{
   throw invalid_argument(error(key, "char*").str());
}

void Parameters::setparam(const string& key, const string& val)
{
   throw invalid_argument(error(key, "string").str());
}

void Parameters::setparam(const string& key, const vector<bool>& val)
{
   throw invalid_argument(error(key, "vector<bool>").str());
}

void Parameters::setparam(const string& key, const vector<int>& val)
{
   throw invalid_argument(error(key, "vector<int>").str());
}

void Parameters::setparam(const string& key, const vector<float>& val)
{
   throw invalid_argument(error(key, "vector<float>").str());
}

void Parameters::setparam(const string& key, const vector<double>& val)
{
   throw invalid_argument(error(key, "vector<double>").str());
}

void Parameters::setparam(const string& key, const vector<char*>& val)
{
   throw invalid_argument(error(key, "vector<char*>").str());
}

void Parameters::setparam(const string& key, const vector<string>& val)
{
   throw invalid_argument(error(key, "vector<string>").str());
}

void Parameters::getparam(const string& key, bool& val)
{
   throw invalid_argument(error(key, "bool").str());
}

void Parameters::getparam(const string& key, int& val)
{
   throw invalid_argument(error(key, "int").str());
}

void Parameters::getparam(const string& key, float& val)
{
   throw invalid_argument(error(key, "float").str());
}

void Parameters::getparam(const string& key, double& val)
{
   throw invalid_argument(error(key, "double").str());
}

void Parameters::getparam(const string& key, char* const val)
{
   throw invalid_argument(error(key, "char*").str());
}

void Parameters::getparam(const string& key, string& val)
{
   throw invalid_argument(error(key, "string").str());
}

void Parameters::getparam(const string& key, vector<bool>& val)
{
   throw invalid_argument(error(key, "vector<bool>").str());
}

void Parameters::getparam(const string& key, vector<int>& val)
{
   throw invalid_argument(error(key, "vector<int>").str());
}

void Parameters::getparam(const string& key, vector<float>& val)
{
   throw invalid_argument(error(key, "vector<float>").str());
}

void Parameters::getparam(const string& key, vector<double>& val)
{
   throw invalid_argument(error(key, "vector<double>").str());
}

void Parameters::getparam(const string& key, vector<char*>& val)
{
   throw invalid_argument(error(key, "vector<char*>").str());
}

void Parameters::getparam(const string& key, vector<string>& val)
{
   throw invalid_argument(error(key, "vector<string>").str());
}
ostringstream& Parameters::error(const string& key, const string& type)
{
   err_strstr.clear();
   err_strstr << "Class (" << class_id << ") : has no parameter '" << key << "' of type " << type << ".";
   return err_strstr;
}

} /* namespace flex_neuralnet */
