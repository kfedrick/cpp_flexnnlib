//
// Created by kfedrick on 9/25/19.
//

#ifndef FLEX_NEURALNET_VECTORCONCATENATOR_H_
#define FLEX_NEURALNET_VECTORCONCATENATOR_H_

#include <valarray>
#include <map>
#include "Vectorizable.h"

namespace flexnnet
{
   class VectorConcatenator
   {
   public:
      virtual const std::valarray<double>& concat(void) const = 0;
   };

}

#endif //FLEX_NEURALNET_VECTORCONCATENATOR_H_
