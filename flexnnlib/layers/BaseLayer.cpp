/*
 * Layer.cpp
 *
 *  Created on: Feb 8, 2014
 *      Author: kfedrick
 */

#include "BaseLayer.h"
#include "URandArrayInitializer.h"

namespace flex_neuralnet
{

const unsigned int BaseLayer::default_history_size = 2;
const unsigned int BaseLayer::initial_layer_input_size = 0;

BaseLayer::BaseLayer(unsigned int sz, const string& _name) :
      NamedObject(_name)
{
   initialize(sz);
}

BaseLayer::BaseLayer(unsigned int sz, const char* _name) :
      NamedObject(_name)
{
   initialize(sz);
}

void BaseLayer::initialize(unsigned int sz)
{
   output_vector_size = sz;
   input_vector_size = initial_layer_input_size;

   resize_history(default_history_size);
   resize(sz);

   transfer_functor_ptr = 0;
   net_input_functor_ptr = 0;
   weight_initializer_ptr = new URandArrayInitializer();

   set_learn_biases(true);
   set_learn_weights(true);
}

BaseLayer::~BaseLayer()
{
   delete transfer_functor_ptr;
   delete net_input_functor_ptr;
   delete weight_initializer_ptr;
}

void BaseLayer::set_transfer_functor(const TransferFunctor* transFunc)
{
   delete transfer_functor_ptr;
   transfer_functor_ptr = transFunc->clone();
}

void BaseLayer::set_netinput_functor(const NetInputFunctor* netinFunc)
{
   delete net_input_functor_ptr;
   net_input_functor_ptr = netinFunc->clone();
}

void BaseLayer::set_weight_initializer(const ArrayInitializer* arrayInit)
{
   delete weight_initializer_ptr;
   weight_initializer_ptr = arrayInit->clone();
}

void BaseLayer::resize(unsigned int sz)
{
   output_vector_size = sz;

   layer_biases.resize(output_vector_size, 0.0);
   resize_weight_array(size(), input_vector_size);

   realloc_history();
}

/*
 * Specify the expected input vector size for this layer
 */
void BaseLayer::resize_input_vector(unsigned int sz)
{
   input_vector_size = sz;

   resize_weight_array(size(), input_vector_size);

   realloc_history();
}

void BaseLayer::resize_weight_array(unsigned int rows, unsigned int cols)
{
   if (rows == 0 || cols == 0)
      return;

   layer_weights.resize(rows, cols);
}

void BaseLayer::resize_history(unsigned int sz)
{
   history_size = sz;

   layer_output.resize(sz);
   net_input.resize(sz);
   layer_output_error.resize(sz);
   layer_input_error.resize(sz);

   dAdN.resize(sz);
   dAdB.resize(sz);
   dNdI.resize(sz);
   dNdW.resize(sz);
   dEdW.resize(sz);
   dEdB.resize(sz);

   d2AdB.resize(sz);
   d2AdN.resize(sz);

   realloc_history();
}

void BaseLayer::realloc_history()
{
   for (unsigned int i = 0; i < history_size; i++)
   {
      if (input_vector_size > 0)
      {
         dNdW.at(i).resize(size(), input_vector_size);
         dNdI.at(i).resize(size(), input_vector_size);
         dEdW.at(i).resize(size(), input_vector_size);

         layer_input_error.at(i).resize(input_vector_size, 0.0);
      }

      dAdN.at(i).resize(size(), size());
      dAdB.at(i).resize(size(), size());

      d2AdB.at(i).resize(size(), size());
      d2AdN.at(i).resize(size(), size());


      layer_output_error.at(i).resize(size(), 0.0);
      layer_output.at(i).resize(size(), 0.0);
      dEdB.at(i).resize(size(), 0.0);
      net_input.at(i).resize(size(), 0.0);
   }
}

void BaseLayer::set_learn_biases(bool _flag)
{
   learn_biases_flag = _flag;
}

void BaseLayer::set_learn_weights(bool _flag)
{
   learn_weights_flag = _flag;
}

bool BaseLayer::is_learn_biases()
{
   return learn_biases_flag;
}

bool BaseLayer::is_learn_weights()
{
   return learn_weights_flag;
}

unsigned int BaseLayer::size() const
{
   return output_vector_size;
}

unsigned int BaseLayer::input_size() const
{
   return input_vector_size;
}

const vector<double>& BaseLayer::operator()(unsigned int timeStep) const
{
   return layer_output.at(timeStep);
}

const vector<double>& BaseLayer::get_error(unsigned int timeStep) const
{
   return layer_output_error.at(timeStep);
}

const vector<double>& BaseLayer::get_input() const
{
   return raw_input;
}

const vector<double>& BaseLayer::get_net_input(unsigned int timeStep) const
{
   return net_input.at(timeStep);
}

const vector<double>& BaseLayer::get_input_error(unsigned int timeStep) const
{
   return layer_input_error.at(timeStep);
}

const vector<double>& BaseLayer::get_biases() const
{
   return layer_biases;
}

const Array<double>& BaseLayer::get_weights() const
{
   return layer_weights;
}

const Array<double>& BaseLayer::get_dNdI(unsigned int timeStep) const
{
   return dNdI.at(timeStep);
}

const Array<double>& BaseLayer::get_dNdW(unsigned int timeStep) const
{
   return dNdW.at(timeStep);
}

const Array<double>& BaseLayer::get_dAdN(unsigned int timeStep) const
{
   return dAdN.at(timeStep);
}

const Array<double>& BaseLayer::get_dAdB(unsigned int timeStep) const
{
   return dAdB.at(timeStep);
}

const vector<double>& BaseLayer::get_dEdB(unsigned int timeStep) const
{
   return dEdB.at(timeStep);
}

const Array<double>& BaseLayer::get_dEdW(unsigned int timeStep) const
{
   return dEdW.at(timeStep);
}

const vector<double>& BaseLayer::get_d2AdN(unsigned int timeStep) const
{
   return d2AdN.at(timeStep);
}

NetInputFunctor* BaseLayer::get_netinput_functor()
{
   return net_input_functor_ptr;
}

TransferFunctor* BaseLayer::get_transfer_functor()
{
   return transfer_functor_ptr;
}

/*
 * Calculate the layers value based on the specified raw input vectors.
 */
void BaseLayer::activate(const vector<double>& inputVec, unsigned int timeStep)
{
   raw_input = inputVec;

   // Calculate net input into transfer function (e.g. weighted sum of input vector over weight matrix)
   (*net_input_functor_ptr)(net_input[timeStep], dNdW[timeStep], dNdI[timeStep],
         inputVec, layer_weights);

   // Calculate the layer output using the transfer function
   (*transfer_functor_ptr)(layer_output[timeStep], dAdN[timeStep], d2AdN[timeStep],
         dAdB[timeStep], net_input[timeStep], layer_biases);
}

/*
 * Accumulate backprop error for this layer
 */
void BaseLayer::backprop(const vector<double>& errorVec, unsigned int timeStep)
{
   vector<double>& errorv = layer_output_error.at(timeStep);
   for (unsigned int ndx = 0; ndx < errorVec.size(); ndx++)
   {
      errorv.at(ndx) += errorVec.at(ndx);
   }

   calc_dEdB(timeStep);
   calc_dEdW(timeStep);
}

/*
 * Backpropagate error through this layer
 */
void BaseLayer::backprop(unsigned int timeStep)
{
   unsigned int layer_size = size();
   unsigned int input_layer_size = input_size();

   vector<double>& errorv = layer_output_error.at(timeStep);
   for (unsigned int in_ndx = 0; in_ndx < input_layer_size; in_ndx++)
   {
      layer_input_error.at(timeStep).at(in_ndx) = 0;

      for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
      {
         /*
          layer_input_error.at(timeStep).at(in_ndx) += errorv.at(out_ndx)
          * dAdI.at(timeStep).at(out_ndx, in_ndx);
          */

         for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
            layer_input_error.at(timeStep).at(in_ndx) += errorv.at(out_ndx)
                  * dAdN.at(timeStep).at(out_ndx, netin_ndx)
                  * dNdI.at(timeStep).at(netin_ndx, in_ndx);
      }
   }
}

void BaseLayer::calc_dEdB(unsigned int timeStep)
{
   unsigned int layer_size = size();

   const vector<double>& errorv = layer_output_error.at(timeStep);
   const Array<double>& curr_dAdB = dAdB.at(timeStep);

   vector<double>& curr_dEdB = dEdB.at(timeStep);

   for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
   {
      curr_dEdB.at(netin_ndx) = 0;
      for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
      {
         curr_dEdB.at(netin_ndx) += errorv.at(out_ndx)
               * curr_dAdB.at(out_ndx, netin_ndx);
      }
   }
}

void BaseLayer::calc_dEdW(unsigned int timeStep)
{
   unsigned int layer_size = size();
   unsigned int layer_input_size = input_size();

   const vector<double>& errorv = layer_output_error.at(timeStep);
   const Array<double>& curr_dAdN = dAdN.at(timeStep);
   const Array<double>& curr_dNdW = dNdW.at(timeStep);

   Array<double>& curr_dEdW = dEdW.at(timeStep);

   double temp;

   /*
    for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
    {
    for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
    {
    curr_dEdW.at(out_ndx, in_ndx) = 0;
    temp = 0;
    for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
    {
    temp += curr_dAdN.at(out_ndx, netin_ndx)
    * curr_dNdW.at(netin_ndx, in_ndx);
    }
    curr_dEdW.at(out_ndx, in_ndx) = errorv.at(out_ndx) * temp;
    }
    }
    */

   for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
   {
      temp = 0;
      for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
         temp += errorv.at(out_ndx) * curr_dAdN.at(out_ndx, netin_ndx);

      for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
         curr_dEdW.at(netin_ndx, in_ndx) = temp
               * curr_dNdW.at(netin_ndx, in_ndx);
   }

}

void BaseLayer::adjust_biases(const vector<double>& deltaB)
{
   if (learn_biases_flag)
   {
      for (unsigned int ndx = 0; ndx < deltaB.size(); ndx++)
         layer_biases.at(ndx) += deltaB.at(ndx);
   }
}

/*
 * Adjust any layer parameters based on the accumulated adjustments and then
 * clears the vector of update values.
 */
void BaseLayer::adjust_weights(const Array<double>& deltaW)
{
   if (learn_weights_flag)
   {
      for (unsigned int row = 0; row < deltaW.rowDim(); row++)
         for (unsigned int col = 0; col < deltaW.colDim(); col++)
            layer_weights.at(row, col) += deltaW.at(row, col);
   }
}

void BaseLayer::clear_error(unsigned int timeStep)
{
   vector<double>& errorv = layer_output_error.at(timeStep);
   for (unsigned int ndx = 0; ndx < errorv.size(); ndx++)
      errorv.at(ndx) = 0;

   for (unsigned int in_ndx = input_vector_size; in_ndx < input_vector_size;
         in_ndx++)
      layer_input_error.at(timeStep).at(in_ndx) = 0;
}

void BaseLayer::init_weights()
{
   init_weights(*weight_initializer_ptr);
}

void BaseLayer::init_weights(const ArrayInitializer& initializer)
{
   initializer(layer_weights);
}

void BaseLayer::set_biases(const vector<double>& bias)
{
   layer_biases = bias;
}

void BaseLayer::set_weights(const Array<double>& weights)
{
   layer_weights = weights;
}

} /* namespace flex_neuralnet */
