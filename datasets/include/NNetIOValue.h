//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_NNETIOVALUE_H_
#define FLEX_NEURALNET_NNETIOVALUE_H_

#include <flexnnet.h>

namespace flexnnet
{
   class NNetIOValue
   {
   public:

      /**
       * Return the coordinate encoded as a NNetIO_Map for use as an
       * input to a neural network.
       *
       * @return
       */
      virtual const ValarrMap& value_map(void) const = 0;

      /**
       * Convert a NNetIO_Map to native encoding.
       *
       * @param _vmap
       * @return
       */
      virtual void parse(const ValarrMap& _vmap) = 0;
   };
}

#endif //FLEX_NEURALNET_NNETIOVALUE_H_
