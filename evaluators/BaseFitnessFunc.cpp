//
// Created by kfedrick on 3/14/21.
//

#include "BaseFitnessFunc.h"

using flexnnet::BaseFitnessFunc;

const std::valarray<double>& BaseFitnessFunc::sqr_diff(const std::valarray<double>& _v1, const std::valarray<double>& _v2)
{
   std::valarray<double> temp;
   static std::valarray<double> sqrdiff;

   temp = _v1 - _v2;
   sqrdiff = temp * temp;

   return sqrdiff;
}

double BaseFitnessFunc::sum_sqr_error(const std::valarray<double>& _v1, const std::valarray<double>& _v2)
{
   return sqr_diff(_v1, _v2).sum();
}
