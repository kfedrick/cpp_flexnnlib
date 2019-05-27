/*
 * NetworkWeightsData.h
 *
 *  Created on: Mar 29, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_NETWORKWEIGHTSDATA_H_
#define FLEX_NEURALNET_NETWORKWEIGHTSDATA_H_

#include "Array.h"
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>

using namespace std;

namespace flexnnet
{

   class LayerWeightsData
   {
   public:
      vector<double> initial_value;
      vector<double> biases;
      Array<double> weights;
   };

   class NetworkWeightsData
   {
   public:
      NetworkWeightsData ();

      const set<string> &keySet () const;
      LayerWeightsData &layer_weights (const string &_id);
      const LayerWeightsData &layer_weights (const string &_id) const;

      LayerWeightsData &new_layer_weights (const string &_id);

      void toFile (const string &_fname);
      void fromFile (const string &_fname);

      void print ();

   private:
      void updateKeySet ();

      void write_layername (fstream &_fs, const string &_name);
      void write_vector (fstream &_fs, const vector<double> &_vec);
      void write_array (fstream &_fs, const Array<double> &_arr);

      void read_layername (fstream &_fs, string &_name);
      void read_vector (fstream &_fs, vector<double> &_vec);
      void read_array (fstream &_fs, Array<double> &_arr);

      mutable set<string> key_set;
      map<string, LayerWeightsData> layer_weights_map;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_NETWORKWEIGHTSDATA_H_ */
