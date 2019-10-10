//
// Created by kfedrick on 10/8/19.
//

#ifndef FLEX_NEURALNET_NETWORKERROR_H_
#define FLEX_NEURALNET_NETWORKERROR_H_

#include <valarray>

namespace flexnnet
{
   struct NetworkError
   {
      double error;
      std::valarray<double> dEdy;
   };
}

#endif //FLEX_NEURALNET_NETWORKERROR_H_
