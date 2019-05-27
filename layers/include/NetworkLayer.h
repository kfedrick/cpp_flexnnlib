//
// Created by kfedrick on 5/12/19.
//

#ifndef FLEX_NEURALNET_NETWORKLAYER_H_
#define FLEX_NEURALNET_NETWORKLAYER_H_

#include "BasicLayer.h"
#include "Pattern.h"

namespace flexnnet
{
   class NetworkLayer;

   class InputEntry
   {
   public:
      InputEntry (NetworkLayer &from, bool recurrent = false);
      InputEntry (unsigned int ndx, const std::vector<double> &inputv);
      virtual ~InputEntry ();

      bool is_external_input () const;
      bool is_recurrent () const;

      unsigned int get_input_pattern_index () const;
      unsigned int get_input_vector_size () const;
      NetworkLayer &get_input_layer () const;

      void set_recurrent (bool val);

   private:
      NetworkLayer* input_layer;

      /*
       * This flag indicates that the connection is from the network input
       */
      bool external_input_flag;

      /*
       * Index into the network input pattern for this connection
       */
      unsigned int input_pattern_index;

      /*
       * Size of the network input vector
       */
      unsigned int input_vector_size;

      /*
       * This must be mutable so we can reassign this flag as we add
       * new connections.
       */
      mutable bool recurrent_connection_flag;
   };
   
   class NetworkLayer : public BasicLayer
   {
   protected:

      /* ********************************************************************
       * Constructors, destructors
       */
      NetworkLayer (unsigned int _sz, const std::string &_name);

   public:
      /* ******************************************************************
       * Public member functions to inteconnect layer for activation and
       * error backpropagation.
       */
       
      void add_input_connection(NetworkLayer& _layer, bool recurrent = false);
      void add_input_connection(const Pattern &ipattern, unsigned int patternNdx);
      const std::vector<double>& coelesce_input(const Pattern &inpattern);
      void backprop_scatter (const std::vector<double> _errorv);

   private:
      unsigned int append_virtual_vector (unsigned int start_ndx, const std::vector<double> &vec);

   private:
      /*
       * Contains on ordered list of references to the network layers that provide input this
       * network layer.
       */
      std::vector<InputEntry> layer_input_map;

      /*
       * Local vector to contain the virtual input vector from the input layers connected to this layer
       */
      std::vector<double> virtual_input_vector;

      /*
       * Local vector to hold the backpropogated error vector for each input layer
       */
      std::vector<std::vector<double> > backprop_error_vector;
   };
}

#endif //FLEX_NEURALNET_NETWORKLAYER_H_
