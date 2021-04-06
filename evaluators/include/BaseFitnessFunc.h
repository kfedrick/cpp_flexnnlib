//
// Created by kfedrick on 3/14/21.
//

#ifndef FLEX_NEURALNET_BASEFITNESSFUNC_H_
#define FLEX_NEURALNET_BASEFITNESSFUNC_H_

#include <valarray>

namespace flexnnet
{
   class BaseFitnessFunc
   {
   public:
      static const std::valarray<double>& sqr_diff(const std::valarray<double>& _v1, const std::valarray<double>& _v2);
      static double sum_sqr_error(const std::valarray<double>& _v1, const std::valarray<double>& _v2);
   };
}
#endif //FLEX_NEURALNET_BASEFITNESSFUNC_H_
