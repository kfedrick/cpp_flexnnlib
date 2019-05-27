/*
 * NetworkWeightsData.cpp
 *
 *  Created on: Mar 29, 2014
 *      Author: kfedrick
 */



#include "NetworkWeightsData.h"

namespace flexnnet
{

   NetworkWeightsData::NetworkWeightsData ()
   {
   }

   const set<string> &NetworkWeightsData::keySet () const
   {
      return key_set;
   }

   LayerWeightsData &NetworkWeightsData::layer_weights (const string &_id)
   {
      return layer_weights_map.at (_id);
   }

   const LayerWeightsData &NetworkWeightsData::layer_weights (const string &_id) const
   {
      return layer_weights_map.at (_id);
   }

   void NetworkWeightsData::updateKeySet ()
   {
      map<string, LayerWeightsData>::iterator iter;
      for (iter = layer_weights_map.begin (); iter != layer_weights_map.end (); iter++)
         key_set.insert (iter->first);
   }

   LayerWeightsData &NetworkWeightsData::new_layer_weights (const string &_id)
   {
      LayerWeightsData &layer_weights = layer_weights_map[_id];
      updateKeySet ();
      return layer_weights;
   }

   void NetworkWeightsData::toFile (const string &_fname)
   {
      fstream fs;
      fs.open (_fname.c_str (), fstream::out | fstream::binary);

      // Iterate through the weights for each layer

      map<string, LayerWeightsData>::iterator iter;
      for (iter = layer_weights_map.begin (); iter != layer_weights_map.end (); iter++)
      {
         const string &layername = iter->first;
         LayerWeightsData &layer_weights_data = iter->second;

         write_layername (fs, layername);
         write_vector (fs, layer_weights_data.initial_value);
         write_vector (fs, layer_weights_data.biases);
         write_array (fs, layer_weights_data.weights);
      }

      fs.close ();
   }

   void NetworkWeightsData::fromFile (const string &_fname)
   {
      fstream fs;
      fs.open (_fname.c_str (), fstream::in | fstream::binary);

      string str;
      unsigned int sz;
      unsigned int rows, cols;
      double val;

      vector<double> initial_value;
      vector<double> biases;

      int max_tries = 3;

      string layer_name;
      LayerWeightsData layer_weights_data;
      for (fs.peek (); !fs.eof (); fs.peek ())
      {
         read_layername (fs, layer_name);
         read_vector (fs, layer_weights_data.initial_value);
         read_vector (fs, layer_weights_data.biases);
         read_array (fs, layer_weights_data.weights);

         cout << "layer weights data for " << layer_name << endl;

         cout << "layer size " << layer_weights_data.initial_value.size () << endl;
         sz = layer_weights_data.initial_value.size ();
         for (unsigned int ndx = 0; ndx < sz; ndx++)
         {
            cout << layer_weights_data.initial_value.at (ndx) << " ";
         }
         cout << endl << "=========" << endl;

         cout << "biases size " << layer_weights_data.biases.size () << endl;
         sz = layer_weights_data.biases.size ();
         for (unsigned int ndx = 0; ndx < sz; ndx++)
         {
            cout << layer_weights_data.biases.at (ndx) << " ";
         }
         cout << endl << "=========" << endl;

         // Print layer weights
         cout << "weights (" << layer_weights_data.weights.rowDim () << "," << layer_weights_data.weights.colDim ()
              << ")" << endl;
         for (int row = 0; row < layer_weights_data.weights.rowDim (); row++)
         {
            for (int col = 0; col < layer_weights_data.weights.colDim (); col++)
               cout << layer_weights_data.weights.at (row, col) << " ";
            cout << endl;
         }
         cout << endl << "**********" << endl;
      }

      fs.close ();

      updateKeySet ();
   }

   void NetworkWeightsData::write_layername (fstream &_fs, const string &_name)
   {

      unsigned int sz = _name.size ();
      cout << "write_layername " << _name.c_str () << " " << sz << endl;

      _fs.write ((char *) &sz, sizeof (unsigned int));

      const char *c_str = _name.c_str ();
      _fs.write (c_str, sz * sizeof (char));
   }

   void NetworkWeightsData::write_vector (fstream &_fs, const vector<double> &_vec)
   {
      unsigned int sz = _vec.size ();
      _fs.write ((char *) &sz, sizeof (unsigned int));

      for (unsigned int ndx = 0; ndx < sz; ndx++)
         _fs.write ((char *) &_vec[ndx], sizeof (double));
   }

   void NetworkWeightsData::write_array (fstream &_fs, const Array<double> &_arr)
   {
      unsigned int rows = _arr.rowDim ();
      unsigned int cols = _arr.colDim ();
      _fs.write ((char *) &rows, sizeof (unsigned int));
      _fs.write ((char *) &cols, sizeof (unsigned int));

      for (unsigned int row_ndx = 0; row_ndx < rows; row_ndx++)
         for (unsigned int col_ndx = 0; col_ndx < cols; col_ndx++)
            _fs.write ((char *) &_arr.at (row_ndx, col_ndx), sizeof (double));
   }

   void NetworkWeightsData::read_layername (fstream &_fs, string &_name)
   {
      unsigned int sz;
      _fs.read ((char *) &sz, sizeof (unsigned int));

      char *c_str = new char[sz + 1];
      _fs.read (c_str, sz * sizeof (char));

      _name.assign (c_str, sz);
      delete[] c_str;
   }

   void NetworkWeightsData::read_vector (fstream &_fs, vector<double> &_vec)
   {
      unsigned int sz;
      _fs.read ((char *) &sz, sizeof (unsigned int));

      cout << "read vector size " << sz << endl;
      _vec.resize (sz);
      for (unsigned int ndx = 0; ndx < sz; ndx++)
         _fs.read ((char *) &_vec.at (ndx), sizeof (double));
   }

   void NetworkWeightsData::read_array (fstream &_fs, Array<double> &_arr)
   {
      unsigned int rows;
      unsigned int cols;
      _fs.read ((char *) &rows, sizeof (unsigned int));
      _fs.read ((char *) &cols, sizeof (unsigned int));

      _arr.resize (rows, cols);
      for (unsigned int row_ndx = 0; row_ndx < rows; row_ndx++)
         for (unsigned int col_ndx = 0; col_ndx < cols; col_ndx++)
            _fs.read ((char *) &_arr.at (row_ndx, col_ndx), sizeof (double));
   }

   void NetworkWeightsData::print ()
   {
      // Iterate through the weights for each layer

      map<string, LayerWeightsData>::iterator iter;
      for (iter = layer_weights_map.begin (); iter != layer_weights_map.end (); iter++)
      {
         cout << "weights for " << iter->first << endl;

         LayerWeightsData &layer_weights_data = iter->second;

         unsigned int sz;

         // Print initial layer values
         cout << "initial values" << endl;
         sz = layer_weights_data.initial_value.size ();
         for (unsigned int ndx = 0; ndx < sz; ndx++)
            cout << layer_weights_data.initial_value[ndx] << " ";
         cout << endl << "************" << endl;

         // Print layer biases
         cout << "biases" << endl;
         sz = layer_weights_data.biases.size ();
         for (unsigned int ndx = 0; ndx < sz; ndx++)
            cout << layer_weights_data.biases[ndx] << " ";
         cout << endl << "************" << endl;

         // Print layer weights
         cout << "weights" << endl;
         for (int row = 0; row < layer_weights_data.weights.rowDim (); row++)
         {
            for (int col = 0; col < layer_weights_data.weights.colDim (); col++)
               cout << layer_weights_data.weights.at (row, col) << " ";
            cout << endl;
         }
         cout << endl << "**********" << endl;
      }
   }

} /* namespace flexnnet */
