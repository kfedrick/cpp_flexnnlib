//
// Created by kfedrick on 5/11/19.
//

#ifndef _DERIVED_H_
#define _DERIVED_H_

#include "Base.h"

namespace flexnnet
{
   class Derived : public Base
   {
   public:
      Derived() : Base()
      {
         using namespace std::placeholders;
         calc_out = std::bind( &Derived::calc_layer_output, this, _1, _2, _3, _4);
         val = 333;
      }

   private:
      int
      calc_layer_output (vector<double> &_output, vector<double> &_netin, const vector<double> &_rawin, const Array<
         double> &_weights)
      {
         return(val);
      }

   private:

   };
}

#endif //_DERIVED_H_
