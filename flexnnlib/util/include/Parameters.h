/*
 * Parameters.h
 *
 *  Created on: Feb 2, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_PARAMETERS_H_
#define FLEX_NEURALNET_PARAMETERS_H_

#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace flex_neuralnet
{

class Parameters
{
public:
   Parameters(const string& id="Parameters");
   virtual ~Parameters();

   virtual void setparam(const string& key, bool val);
   virtual void setparam(const string& key, int val);
   virtual void setparam(const string& key, float val);
   virtual void setparam(const string& key, double val);
   virtual void setparam(const string& key, const char* val);
   virtual void setparam(const string& key, const string& val);

   virtual void setparam(const string& key, const vector<bool>& val);
   virtual void setparam(const string& key, const vector<int>& val);
   virtual void setparam(const string& key, const vector<float>& val);
   virtual void setparam(const string& key, const vector<double>& val);
   virtual void setparam(const string& key, const vector<char*>& val);
   virtual void setparam(const string& key, const vector<string>& val);

   virtual void getparam(const string& key, bool& val);
   virtual void getparam(const string& key, int& val);
   virtual void getparam(const string& key, float& val);
   virtual void getparam(const string& key, double& val);
   virtual void getparam(const string& key, char* const val);
   virtual void getparam(const string& key, string& val);

   virtual void getparam(const string& key, vector<bool>& val);
   virtual void getparam(const string& key, vector<int>& val);
   virtual void getparam(const string& key, vector<float>& val);
   virtual void getparam(const string& key, vector<double>& val);
   virtual void getparam(const string& key, vector<char*>& val);
   virtual void getparam(const string& key, vector<string>& val);

protected:
   string class_id;

private:
   ostringstream err_strstr;
   ostringstream& error(const string& key, const string& type);
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_PARAMETERS_H_ */
