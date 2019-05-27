//
// Created by kfedrick on 5/12/19.
//

#include "NetworkLayer.h"

using namespace flexnnet;

InputEntry::InputEntry (NetworkLayer &from, bool recurrent) : input_layer(&from)
{
   external_input_flag = false;
   recurrent_connection_flag = recurrent;
}

InputEntry::InputEntry (unsigned int ndx, const std::vector<double> &inputv) : input_layer(0)
{
   external_input_flag = true;
   input_vector_size = inputv.size ();
   input_pattern_index = ndx;
}

InputEntry::~InputEntry ()
{
   // TODO Auto-generated destructor stub
}

bool InputEntry::is_external_input () const
{
   return external_input_flag;
}

bool InputEntry::is_recurrent () const
{
   return recurrent_connection_flag;
}

unsigned int InputEntry::get_input_pattern_index () const
{
   return input_pattern_index;
}

unsigned int InputEntry::get_input_vector_size () const
{
   return input_vector_size;
}

NetworkLayer &InputEntry::get_input_layer () const
{
   return *input_layer;
}

void InputEntry::set_recurrent (bool val)
{
   recurrent_connection_flag = val;
}

NetworkLayer::NetworkLayer (unsigned int _sz, const std::string &_name) : BasicLayer(_sz, _name)
{

}

void NetworkLayer::add_input_connection(NetworkLayer& _layer, bool _recurrent)
{
   layer_input_map.push_back (InputEntry (_layer, _recurrent));
   virtual_input_vector.resize (virtual_input_vector.size () + _layer.size ());
   backprop_error_vector.push_back (std::vector<double> (_layer.size ()));
}

void NetworkLayer::add_input_connection(const Pattern &ipattern, unsigned int patternNdx)
{
   layer_input_map.push_back (InputEntry (patternNdx, ipattern.at (patternNdx)));
   virtual_input_vector.resize (virtual_input_vector.size () + ipattern.at (patternNdx).size ());
   backprop_error_vector.push_back (std::vector<double> (ipattern.at (patternNdx).size ()));
}

const std::vector<double>& NetworkLayer::coelesce_input(const Pattern &inpattern)
{
   int sz = virtual_input_vector.size ();

   unsigned int virtual_ndx = 0;
   for (unsigned int map_ndx = 0; map_ndx < layer_input_map.size (); map_ndx++)
   {
      InputEntry &conn = layer_input_map[map_ndx];

      if (conn.is_external_input ())
      {
         const std::vector<double> &inputv = inpattern.at (conn.get_input_pattern_index ());
         virtual_ndx = append_virtual_vector (virtual_ndx, inputv);
      }
      else
      {
         const NetworkLayer &in_layer = conn.get_input_layer ();

         const std::vector<double> &layer_outputv = in_layer ();
         virtual_ndx = append_virtual_vector (virtual_ndx, layer_outputv);
      }
   }

   return virtual_input_vector;
}

void NetworkLayer::backprop_scatter (const std::vector<double> _errorv)
{
   unsigned int sz = _errorv.size ();

   // TODO - range checking, clean up implementation
   int errv_ndx = 0;
   for (unsigned int map_ndx = 0; map_ndx < layer_input_map.size (); map_ndx++)
   {
      InputEntry &conn = layer_input_map[map_ndx];
      unsigned int backprop_errorv_sz;

      if (conn.is_external_input ())
         backprop_errorv_sz = conn.get_input_vector_size ();
      else
      {
         const NetworkLayer &in_layer = conn.get_input_layer ();
         backprop_errorv_sz = in_layer.size ();
      }

      for (unsigned int backprop_errorv_ndx = 0; backprop_errorv_ndx < backprop_errorv_sz; backprop_errorv_ndx++)
         backprop_error_vector.at (map_ndx).at (backprop_errorv_ndx) = _errorv.at (errv_ndx++);

      // Tell layer to accumulate the new errorv
      if (!conn.is_external_input ())
      {
         NetworkLayer &in_layer = conn.get_input_layer ();
         in_layer.accumulate_error (backprop_error_vector[map_ndx]);
      }
   }
}


unsigned int NetworkLayer::append_virtual_vector (unsigned int start_ndx, const std::vector<double> &vec)
{
   unsigned int virtual_ndx = start_ndx;
   for (unsigned int ndx = 0; ndx < vec.size (); ndx++)
      virtual_input_vector.at (virtual_ndx++) = vec.at (ndx);
   return virtual_ndx;
}

