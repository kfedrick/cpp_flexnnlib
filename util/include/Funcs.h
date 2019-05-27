//
// Created by kfedrick on 5/11/19.
//

#ifndef _FUNCS_H_
#define _FUNCS_H_

#include "Array.h"

namespace flexnnet
{
   class Funcs
   {
   public:

      static int
      calc_layer_output (vector<double> &_output, vector<double> &_netin, const vector<double> &_rawin, const Array<
         double> &_weights)
      {
         return(_output.size());
      }

      static int
      calc_layer_derivatives (Array<double> &_dAdW, vector<double> &_dADI, const vector<double> &_output, const vector<
         double> &_netin, const vector<double> &_rawin, const Array<double> &weights)
      {
         return(_output.size());
      }

   };
}

#endif //_FUNCS_H_
