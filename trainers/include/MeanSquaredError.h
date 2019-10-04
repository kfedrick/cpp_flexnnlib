//
// Created by kfedrick on 10/3/19.
//

#ifndef FLEX_NEURALNET_MEANSQUAREDERROR_H_
#define FLEX_NEURALNET_MEANSQUAREDERROR_H_

#include <valarray>

namespace flexnnet
{
   class MeanSquaredError
   {
      double operator()(std::valarray<double>& _target, std::valarray<double>& _observed);
   };

   inline
   double MeanSquaredError::operator()(std::valarray<double>& _target, std::valarray<double>& _observed)
   {
      return 0;
   }

}

#endif //FLEX_NEURALNET_MEANSQUAREDERROR_H_
