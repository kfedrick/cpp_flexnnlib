/*
 * Layer.h
 *
 *  Created on: Feb 8, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_BASELAYER_H_
#define FLEX_NEURALNET_BASELAYER_H_

#include "NamedObject.h"
#include "TransferFunctor.h"
#include "NetInputFunctor.h"
#include "ArrayInitializer.h"

#include <vector>
#include <stdexcept>
#include <iostream>

using namespace std;

namespace flex_neuralnet
{

class BaseLayer: public flex_neuralnet::NamedObject
{
   /* ********************************************************************
    *    Static const class values
    */
protected:
   static const unsigned int default_history_size;
   static const unsigned int initial_layer_input_size;

public:

   /* ********************************************************************
    *    Constructors, destructors and configuration methods
    */
   BaseLayer(unsigned int sz, const char* name  = "BaseLayer");
   BaseLayer(unsigned int sz, const string& name);
   virtual ~BaseLayer();

   /*
    * Set weight initializer function object
    */
   void set_weight_initializer(const ArrayInitializer* arrayInit);

   /*
    * Resize the number of neurons in the layer
    */
   virtual void resize(unsigned int sz);

   /*
    * Specify the expected input vector size for this layer
    */
   virtual void resize_input_vector(unsigned int sz);

   /*
    * Resize the activation history capacity for the layer
    */
   virtual void resize_history(unsigned int sz = 2);


   /* ********************************************************************
    *    Getter methods
    */

   bool is_learn_biases();

   bool is_learn_weights();

   /*
    * Return the length of the layers output vector
    */
   virtual unsigned int size() const;

   /*
    * Return the length of the layers input vector
    */
   virtual unsigned int input_size() const;

   /*
    * Return the current value of the network layer as a vector
    */
   virtual const vector<double>& operator()(unsigned int timeStep = 1) const;

   virtual const vector<double>& get_net_input(unsigned int timeStep = 1) const;

   /*
    * Get the error vector for the layer output vector at the specified times step.
    */
   virtual const vector<double>& get_error(unsigned int timeStep = 1) const;

   /*
    * Get the backpropogated error signal for the layer input
    */
   virtual const vector<double>& get_input_error(unsigned int timeStep = 1) const;

   const vector<double>& get_biases() const;

   const Array<double>& get_weights() const;


   const Array<double>& get_dNdI(unsigned int timeStep = 1) const;

   const Array<double>& get_dNdW(unsigned int timeStep = 1) const;

   const Array<double>& get_dAdN(unsigned int timeStep = 1) const;

   const Array<double>& get_dAdB(unsigned int timeStep = 1) const;

   const vector<double>& get_dEdB(unsigned int timeStep = 1) const;

   const Array<double>& get_dEdW(unsigned int timeStep = 1) const;


   NetInputFunctor* get_netinput_functor();
   TransferFunctor* get_transfer_functor();


   /* ********************************************************************
    *    Layer operational methods
    */


   /*
    * Calculate the value of the layer neuron vector based on the specified raw input vectors.
    */
   virtual void activate(const vector<double>& inputVec, unsigned int timeStep = 1);

   /*
    * Accumulate backprop error for this layer
    */
   virtual void backprop(const vector<double>& errorVec, unsigned int timeStep = 1);

   /*
    * Backpropogate error through this layer
    */
   virtual void backprop(unsigned int timeStep = 1);

   virtual void adjust_biases(const vector<double>& deltaB);

   virtual void adjust_weights(const Array<double>& deltaW);

   virtual void clear_error(unsigned int timeStep = 1);

   /* ********************************************************************
    *    Housekeeping functions
    */

   void init_weights();

   void init_weights(const ArrayInitializer& initializer);

   void set_learn_biases(bool flag);

   void set_learn_weights(bool flag);

   void set_biases(const vector<double>& bias);

   void set_weights(const Array<double>& weights);


protected:
   /* *************************************************
    *    Set policy objects
    */

   /*
    * Set net input function object
    */
   void set_netinput_functor(const NetInputFunctor* netinFunc);

   /*
    * Set transfer function object
    */
   void set_transfer_functor(const TransferFunctor* transFunc);



private:

   void initialize(unsigned int sz);

   /*
    * Resize the layers weight array
    */
   void resize_weight_array(unsigned int rows, unsigned int cols);

   void realloc_history();

   void calc_dEdB(unsigned int timeStep = 1);

   void calc_dEdW(unsigned int timeStep = 1);

private:

   /* *************************************************
    *    Policy objects
    */

   /*
    * Net input function object
    */
   NetInputFunctor* net_input_functor_ptr;

   /*
    * Transfer function object
    */
   TransferFunctor* transfer_functor_ptr;

   /*
    * Network weight initializer function object
    */
   ArrayInitializer* weight_initializer_ptr;

   bool learn_biases_flag;
   bool learn_weights_flag;


   /* *************************************************
    *    Core data objects
    */

   unsigned int output_vector_size;
   unsigned int input_vector_size;
   unsigned int history_size;

   // History of layer value vectors (for recurrent networks)
   vector< vector<double> > layer_output;

   // History of layer net input vectors (for recurrent networks)
   vector< vector<double> > net_input;

   // History of error vectors for the layer output
   vector< vector<double> > layer_output_error;

   vector< vector<double> > layer_input_error;

   vector<double> layer_biases;
   Array<double> layer_weights;

   // The derivative of the net input with respect to the layer weights history
   vector< Array<double> > dNdW;

   // The derivative of the net input with respect to the raw input vector history
   vector< Array<double> > dNdI;

   // The derivative of the transfer function output with respect to the net input history
   vector< Array<double> > dAdN;

   // History of derivative of the layer output with respect to the biases
   vector< Array<double> > dAdB;

   // History of the derivative of the layer output with respect to the i'th bias
   vector< vector<double> > dEdB;

   // History of the derivative of the layer output vector with respect to W(i,j)
   vector< Array<double> > dEdW;
};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_BASELAYER_H_ */
