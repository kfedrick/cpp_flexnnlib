/*
 * HvMap.cpp
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#include "HvMap.h"

#include <iostream>

using namespace std;

namespace flexnnet
{

   HvMap::HvMap (const ConnectionMap &_connMap, const map<string, vector<double>> &_ryMap) : Ry_map (_ryMap)
   {
      const vector<ConnectionEntry> &conne = _connMap.get_input_connections ();

      layer_input_map = conne;
   }

   HvMap::~HvMap ()
   {
//   cout << "HvMap::~HvMap()" << endl;
      //layer_input_map.clear();
   }

   int HvMap::size () const
   {
      return virtual_input_vector.size ();
   }

   const vector<double> &HvMap::operator() ()
   {
      int sz = virtual_input_vector.size ();

      unsigned int virtual_ndx = 0;
      for (unsigned int map_ndx = 0; map_ndx < layer_input_map.size (); map_ndx++)
      {
         ConnectionEntry &conn = layer_input_map[map_ndx];

         if (conn.is_input_connection ())
         {
            vector<double> inputv (conn.get_input_vector_size (), 1.0);
            virtual_ndx = append_virtual_vector (virtual_ndx, inputv);
         }
         else
         {
            const BaseLayer &in_layer = conn.get_input_layer ();
            const string &aname = in_layer.name ();

            const vector<double> &Ry = Ry_map.at (aname);
            virtual_ndx = append_virtual_vector (virtual_ndx, Ry);
         }
      }

      return virtual_input_vector;
   }

   const vector<vector<double> > &HvMap::get_error (const vector<double> &errorv)
   {
      this->backprop_scatter (errorv);

      return backprop_error_vector;
   }

   const vector<vector<double> > &HvMap::get_error (unsigned int timeStep)
   {
      const vector<double> &errorv = target_layer->get_input_error (timeStep);
      this->backprop_scatter (errorv);

      return backprop_error_vector;
   }

/*
 * Scatters the coalesced backprop error vector into sub-vectors for each input layer.
 */
   void HvMap::backprop_scatter (const vector<double> &errorv)
   {
      unsigned int sz = errorv.size ();

      // TODO - range checking
      int errv_ndx = 0;
      for (unsigned int map_ndx = 0; map_ndx < layer_input_map.size (); map_ndx++)
      {
         ConnectionEntry &conn = layer_input_map[map_ndx];
         unsigned int backprop_errorv_sz;

         if (conn.is_input_connection ())
            backprop_errorv_sz = conn.get_input_vector_size ();
         else
         {
            const BaseLayer &in_layer = conn.get_input_layer ();
            backprop_errorv_sz = in_layer.size ();
         }

         for (unsigned int backprop_errorv_ndx = 0; backprop_errorv_ndx < backprop_errorv_sz; backprop_errorv_ndx++)
            backprop_error_vector.at (map_ndx).at (backprop_errorv_ndx) = errorv.at (errv_ndx++);
      }
   }

   void HvMap::clear_error ()
   {
      for (int map_ndx = 0; map_ndx < layer_input_map.size (); map_ndx++)
      {
         vector<double> &bp_errorv = backprop_error_vector.at (map_ndx);
         for (unsigned int backprop_errorv_ndx = 0; backprop_errorv_ndx < backprop_error_vector.at (map_ndx).size ();
              backprop_errorv_ndx++)
            bp_errorv.at (backprop_errorv_ndx) = 0;
      }
   }

   vector<ConnectionEntry> &HvMap::get_input_connections ()
   {
      return layer_input_map;
   }

   int HvMap::input_map_size () const
   {
      return layer_input_map.size ();
   }

   void HvMap::connect (BaseLayer &layer, bool recurrent)
   {
      layer_input_map.push_back (ConnectionEntry (layer, recurrent));
      virtual_input_vector.resize (virtual_input_vector.size () + layer.size ());
      backprop_error_vector.push_back (vector<double> (layer.size ()));
   }

   void HvMap::connect (const Pattern &ipattern, unsigned int patternNdx)
   {
      layer_input_map.push_back (ConnectionEntry (patternNdx, ipattern.at (patternNdx)));
      virtual_input_vector.resize (virtual_input_vector.size () + ipattern.at (patternNdx).size ());
      backprop_error_vector.push_back (vector<double> (ipattern.at (patternNdx).size ()));
   }

   unsigned int HvMap::append_virtual_vector (unsigned int start_ndx, const vector<double> &vec)
   {
      unsigned int virtual_ndx = start_ndx;
      for (unsigned int ndx = 0; ndx < vec.size (); ndx++)
         virtual_input_vector.at (virtual_ndx++) = vec.at (ndx);
      return virtual_ndx;
   }

} /* namespace flexnnet */
