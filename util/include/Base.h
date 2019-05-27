//
// Created by kfedrick on 5/11/19.
//

#ifndef _BASE_H_
#define _BASE_H_

#include <functional>
#include "Funcs.h"

namespace flexnnet
{
   class Base
   {
   public:
      Base()
      {
         printf("Base() Constructor\n");
         calc_layer_derivatives = Funcs::calc_layer_derivatives;

         val = 666;
      }

      void doit()
      {
         vector<double> out, rawin, netin, dAdI;
         Array<double> weights, dAdW;

         printf("doit 0 out = %d\n", Funcs::calc_layer_derivatives (dAdW, dAdI, out, netin, rawin, weights));

         printf("doit out = %d\n", calc_layer_derivatives(dAdW, dAdI, out, netin, rawin, weights));
      }

      void calcit()
      {
         vector<double> out, rawin, netin, dAdI;
         Array<double> weights, dAdW;
         printf("calcit out = %d\n", calc_out(out, netin, rawin, weights));
      }

   protected:
      int val;

   protected:
      std::function<int(vector<double> &_output, vector<double> &_netin, const vector<double> &_rawin, const Array<
         double> &_weights)> calc_out;

   private:
      std::function<int(Array<double> &_dAdW, vector<double> &_dAdI, const vector<double> &_output, const vector<
         double> &_netin, const vector<double> &_rawin, const Array<double> &weights)> calc_layer_derivatives;
   };
}

#endif //_BASE_H_
