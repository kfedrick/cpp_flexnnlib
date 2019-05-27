/*
 * HvMap.h
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_HVMAP_H_
#define FLEX_NEURALNET_HVMAP_H_

#include "ConnectionMap.h"
#include "BaseLayer.h"
#include "Pattern.h"
#include <vector>
#include <map>

using namespace std;

namespace flexnnet
{

   class HvMap
   {

   public:

      /* **********************************************************************
       *    Constructors, Destructors
       * **********************************************************************
       */

      HvMap (const ConnectionMap &_connMap, const map<string, vector<double>> &_ryMap);
      virtual ~HvMap ();

      /* ***********************************************************************
       *    Accessor functons
       * ***********************************************************************
       */

      /*
       * Returns the size of the virtual input vector for this layer
       */
      int size () const;

      /*
       * Returns the number of layers providing input to this layer.
       */
      int input_map_size () const;

      const vector<double> &operator() ();

      const vector<vector<double> > &get_error (const vector<double> &errorv);

      const vector<vector<double> > &get_error (unsigned int timeStep = 1);

      /*
       * Clear the input error vectors
       */
      void clear_error ();

      /* *************************************************************************
       *    BaseLayer topology management functions
       * *************************************************************************
       */
      void connect (BaseLayer &layer, bool recurrent = false);
      void connect (const Pattern &ipattern, unsigned int patternNdx);

      vector<ConnectionEntry> &get_input_connections ();

   private:

      /* ***********************************************************************
       *    Activation functions
       * ***********************************************************************
       */

      /*
       * Scatters the coalesced backprop error vector into sub-vectors for each input layer.
       * Pull the error vector from the target layer for this map.
       */
      void backprop_scatter (const vector<double> &errorv);

      unsigned int append_virtual_vector (unsigned int start_ndx, const vector<double> &vec);

      BaseLayer *target_layer;

      /*
       * Contains on ordered list of references to the network layers that provide input this
       * network layer.
       */
      vector<ConnectionEntry> layer_input_map;

      /*
       * Local vector to contain the virtual input vector from the input layers connected to this layer
       */
      vector<double> virtual_input_vector;

      /*
       * Local vector to hold the backpropogated error vector for each input layer
       */
      vector<vector<double> > backprop_error_vector;

      const map<string, vector<double>> &Ry_map;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_HVMAP_H_ */
