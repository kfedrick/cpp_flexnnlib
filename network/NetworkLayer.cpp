//
// Created by kfedrick on 4/9/21.
//

#include "NetworkLayer.h"

using flexnnet::NetworkLayer;
using flexnnet::ValarrMap;

NetworkLayer::NetworkLayer(bool _is_output) : BaseNetworkLayer()
{
   output_layer_flag = _is_output;
}

NetworkLayer::NetworkLayer(const NetworkLayer& _layer) : BaseNetworkLayer(_layer)
{
   //std::cout << "NetworkLayer::NetworkLayer(const NetworkLayer&) ENTRY\n";

   layer_state = _layer.layer_state;
   output_layer_flag = _layer.output_layer_flag;
   external_input_fields = _layer.external_input_fields;
   //activation_connections = _layer.activation_connections;
   //backprop_connections = _layer.backprop_connections;
}

NetworkLayer& NetworkLayer::operator=(const NetworkLayer& _layer)
{
   //std::cout << "NetworkLayer::operator=(const NetworkLayer&) ENTRY\n";

   layer_state = _layer.layer_state;
   output_layer_flag = _layer.output_layer_flag;
   external_input_fields = _layer.external_input_fields;
   //activation_connections = _layer.activation_connections;
   //backprop_connections = _layer.backprop_connections;

   return *this;
}

NetworkLayer::~NetworkLayer()
{}

void NetworkLayer::concat_inputs(const ValarrMap& _externin, std::valarray<double>& _invec)
{
   size_t virtual_ndx = 0;
   //std::cout << "NetworkLayer.concat_inputs()\n" << std::flush;

   // First append external input fields to virtual input vector
   for (auto& inputrec : external_input_fields)
   {
      //std::cout << "NetworkLayer.concat_inputs() 0 " << inputrec.field()  << "\n" << std::flush;
      //std::cout << "NetworkLayer.concat_inputs() map size " << _externin.size()  << "\n" << std::flush;
      //std::cout << "NetworkLayer.concat_inputs() map first name " << _externin.begin()->first  << "\n" << std::flush;

      const std::valarray<double>& xinputv = _externin.at(inputrec.field());

      append_virtual_vector(xinputv,virtual_ndx,_invec);
   }

   // Append input layer values to the virtual input vector for this layer.
   for (size_t i = 0; i < activation_connections.size(); i++)
   {
      LayerConnRecord& conn = activation_connections[i];

      const NetworkLayer& source_layer = conn.layer();
      const std::valarray<double>& layer_outputv = source_layer.value();

      append_virtual_vector(layer_outputv,virtual_ndx,_invec);
   }

   //std::cout << "NetworkLayer.concat_inputs() EXIT\n" << std::flush;
}

const ValarrMap& NetworkLayer::marshal_inputs(const ValarrMap& _externin)
{
   // First append external input fields
   for (auto& inputrec : external_input_fields)
      input_map[inputrec.field()] = _externin.at(inputrec.field());

   // Append input layer values to the virtual input vector for this layer.
   for (size_t i = 0; i < activation_connections.size(); i++)
   {
      LayerConnRecord& conn = activation_connections[i];
      const NetworkLayer& source_layer = conn.layer();

      input_map[source_layer.name()] = source_layer.value();
   }

   return input_map;
}

/**
 * Marshal the external and back-propagated errors to calculate
 * the cumulative external error vector for this layer.
 *
 * @param _externerr
 * @return
 */
void NetworkLayer::gather_error(const ValarrMap& _externerr, std::valarray<double>& _dEdy)
{
   const std::string& id = name();
   _dEdy = 0;

   // If this is an output layer first set the layer error to the
   // external error.
   if (is_output_layer())
      _dEdy = _externerr.at(id);

   /*
    * Iterate through all of the back-propagation connections and
    * accumulate the backprop error, dEdx, from each upstream layer.
    */
   for (size_t i = 0; i < backprop_connections.size(); i++)
   {
      LayerConnRecord& conn = backprop_connections[i];
      const NetworkLayer& downstream_layer = conn.layer();

      _dEdy += downstream_layer.input_error_map.at(id);
   }
}

void NetworkLayer::scatter_input_error(const std::valarray<double>& _dEdx)
{
   unsigned int ierr_ndx = 0;
   for (auto& inputrec : external_input_fields)
   {
      std::valarray<double>& src_errorv = external_input_error_map[inputrec.field()];

      // Copy slice of full error vector to error for this source layer
      unsigned int src_sz = src_errorv.size();
      for (unsigned int src_ndx = 0; src_ndx < src_sz; src_ndx++)
         src_errorv[src_ndx] = _dEdx[ierr_ndx++];
   }

   for (size_t i = 0; i < activation_connections.size(); i++)
   {
      LayerConnRecord& conn = activation_connections[i];
      const NetworkLayer& src_layer = conn.layer();
      std::valarray<double>& src_errorv = input_error_map[src_layer.name()];

      // Copy slice of full error vector to error for this source layer
      unsigned int src_sz = src_layer.size();
      for (unsigned int src_ndx = 0; src_ndx < src_sz; src_ndx++)
         src_errorv[src_ndx] = _dEdx[ierr_ndx++];
   }
}

size_t NetworkLayer::append_virtual_vector(const std::valarray<double>& _srcvec, size_t& _vindex, std::valarray<double>& _tgtvec)
{
   size_t layer_sz = _srcvec.size();
   for (size_t ndx = 0; ndx < layer_sz; ndx++)
      _tgtvec[_vindex++] = _srcvec[ndx];
}

