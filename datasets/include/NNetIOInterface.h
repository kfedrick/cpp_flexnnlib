//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_NNETIOINTERFACE_H_
#define FLEX_NEURALNET_NNETIOINTERFACE_H_

#include <flexnnet.h>

namespace flexnnet
{
   class NNetIOInterface
   {
   public:

      virtual size_t size(void) const = 0;

      /**
       * Return the coordinate encoded as a valarray<double> for use as an
       * input to a neural network.
       * @return
       */
      virtual const std::valarray<double>& value(void) const = 0;

      /**
       * Return the coordinate encoded as a VararrMap for use as an
       * input to a neural network.
       *
       * @return
       */
      virtual const ValarrMap& value_map(void) const = 0;

      /**
       * Convert a ValarrMap to native encoding.
       *
       * @param _vmap
       * @return
       */
      virtual void parse(const ValarrMap& _vmap) = 0;
   };
}

#endif //FLEX_NEURALNET_NNETIOINTERFACE_H_
