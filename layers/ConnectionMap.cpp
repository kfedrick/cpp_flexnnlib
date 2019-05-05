/*
 * ConnectionMap.cpp
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#include "ConnectionMap.h"

#include <iostream>

using namespace std;

namespace flex_neuralnet
{

ConnectionEntry::ConnectionEntry(BaseLayer& from, bool recurrent)
{
   input_connection_flag = false;

   input_layer = &from;
   recurrent_connection_flag = recurrent;
}

ConnectionEntry::ConnectionEntry(unsigned int ndx, const vector<double>& inputv)
{
   input_connection_flag = true;
   input_layer = 0;
   input_vector_size = inputv.size();
   input_pattern_index = ndx;
}

ConnectionEntry::~ConnectionEntry()
{
   // TODO Auto-generated destructor stub
}

bool ConnectionEntry::is_input_connection() const
{
   return input_connection_flag;
}

bool ConnectionEntry::is_recurrent() const
{
   return recurrent_connection_flag;
}

unsigned int ConnectionEntry::get_input_pattern_index() const
{
   return input_pattern_index;
}

unsigned int ConnectionEntry::get_input_vector_size() const
{
   return input_vector_size;
}

BaseLayer& ConnectionEntry::get_input_layer() const
{
   return *input_layer;
}

void ConnectionEntry::set_recurrent(bool val)
{
   recurrent_connection_flag = val;
}




ConnectionMap::ConnectionMap()
{
   target_layer = 0;
}

ConnectionMap::ConnectionMap(BaseLayer& target)
{
   target_layer = &target;
}

ConnectionMap::~ConnectionMap()
{
//   cout << "ConnectionMap::~ConnectionMap()" << endl;
   //layer_input_map.clear();
}

int ConnectionMap::size() const
{
   return virtual_input_vector.size();
}

const vector<double>& ConnectionMap::operator()(const Pattern& inpattern, unsigned int timeStep, unsigned int closedLoopStep)
{
   int sz = virtual_input_vector.size();

   unsigned int virtual_ndx = 0;
   for (unsigned int map_ndx = 0; map_ndx < layer_input_map.size(); map_ndx++)
   {
      ConnectionEntry& conn = layer_input_map[map_ndx];

      if (conn.is_input_connection())
      {
         const vector<double>& inputv = inpattern.at(conn.get_input_pattern_index());
         virtual_ndx = append_virtual_vector( virtual_ndx, inputv );
      }
      else
      {
         const BaseLayer& in_layer = conn.get_input_layer();

         unsigned int ilayer_timestep = timeStep;
         if (conn.is_recurrent() && closedLoopStep == 0)
            ilayer_timestep--;

         const vector<double>& layer_outputv = in_layer(ilayer_timestep);
         virtual_ndx = append_virtual_vector( virtual_ndx, layer_outputv );
      }
   }

   return virtual_input_vector;
}

const vector< vector<double> >& ConnectionMap::get_error(const vector<double>& errorv)
{
   this->backprop_scatter(errorv);

   return backprop_error_vector;
}

const vector< vector<double> >& ConnectionMap::get_error(unsigned int timeStep)
{
   const vector<double>& errorv = target_layer->get_input_error(timeStep);
   this->backprop_scatter(errorv);

   return backprop_error_vector;
}

/*
 * Scatters the coalesced backprop error vector into sub-vectors for each input layer.
 */
void ConnectionMap::backprop_scatter(const vector<double>& errorv)
{
   unsigned int sz = errorv.size();

   // TODO - range checking
   int errv_ndx = 0;
   for (unsigned int map_ndx = 0; map_ndx < layer_input_map.size(); map_ndx++)
   {
      ConnectionEntry& conn = layer_input_map[map_ndx];
      unsigned int backprop_errorv_sz;

      if (conn.is_input_connection())
         backprop_errorv_sz = conn.get_input_vector_size();
      else
      {
         const BaseLayer& in_layer = conn.get_input_layer();
         backprop_errorv_sz = in_layer.size();
      }

      for (unsigned int backprop_errorv_ndx=0; backprop_errorv_ndx < backprop_errorv_sz; backprop_errorv_ndx++)
          backprop_error_vector.at(map_ndx).at(backprop_errorv_ndx) = errorv.at(errv_ndx++);
   }
}

void ConnectionMap::clear_error()
{
   for (int map_ndx = 0; map_ndx < layer_input_map.size(); map_ndx++)
   {
      vector<double>& bp_errorv = backprop_error_vector.at(map_ndx);
      for (unsigned int backprop_errorv_ndx=0; backprop_errorv_ndx < backprop_error_vector.at(map_ndx).size(); backprop_errorv_ndx++)
          bp_errorv.at(backprop_errorv_ndx) = 0;
   }
}


const vector<ConnectionEntry>& ConnectionMap::get_input_connections() const
{
   return layer_input_map;
}

vector<ConnectionEntry>& ConnectionMap::get_input_connections()
{
   return layer_input_map;
}

int ConnectionMap::input_map_size() const
{
   return layer_input_map.size();
}

void ConnectionMap::connect(BaseLayer& layer, bool recurrent)
{
   layer_input_map.push_back( ConnectionEntry(layer, recurrent) );
   virtual_input_vector.resize( virtual_input_vector.size() + layer.size() );
   backprop_error_vector.push_back(vector<double>(layer.size()));
}

void ConnectionMap::connect(const Pattern& ipattern, unsigned int patternNdx)
{
   layer_input_map.push_back( ConnectionEntry(patternNdx, ipattern.at(patternNdx)) );
   virtual_input_vector.resize( virtual_input_vector.size() + ipattern.at(patternNdx).size() );
   backprop_error_vector.push_back(vector<double>(ipattern.at(patternNdx).size()));
}

unsigned int ConnectionMap::append_virtual_vector(unsigned int start_ndx, const vector<double>& vec)
{
   unsigned int virtual_ndx = start_ndx;
   for (unsigned int ndx = 0; ndx < vec.size(); ndx++)
         virtual_input_vector.at(virtual_ndx++) = vec.at(ndx);
   return virtual_ndx;
}

} /* namespace flex_neuralnet */
