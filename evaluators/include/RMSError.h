//
// Created by kfedrick on 10/3/19.
//

#ifndef FLEX_NEURALNET_RMSERROR_H_
#define FLEX_NEURALNET_RMSERROR_H_

#include <valarray>
#include <util/include/NetworkError.h>

namespace flexnnet
{
   template <typename _Type>
   class RMSError
   {
   public:
      NetworkError error(const _Type& _target, const _Type& _observed);
   };

   template <typename _Type>
   NetworkError RMSError<_Type>::error(const _Type& _target, const _Type& _observed)
   {
      std::valarray<double> dEdy(_target.vectorize() - _observed.vectorize());
      dEdy = dEdy * dEdy;

      // Return square root of the mean squared error (RMSError)
      return { sqrt(dEdy.sum()/dEdy.size()), .dEdy = dEdy };
   }

}

#endif //FLEX_NEURALNET_RMSERROR_H_
