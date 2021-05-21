//
// Created by kfedrick on 3/7/21.
//

#ifndef FLEX_NEURALNET_STATEVIEW_H_
#define FLEX_NEURALNET_STATEVIEW_H_

#include <flexnnet.h>

namespace flexnnet
{
   /**
    * StateView defines the public interface functions for converting
    * to and from the domain state model and representations suitable
    * as input to a neural network (a real-valued vector or a map of
    * string to real-valued vectors).
    */
   class StateView
   {
   public:

      /**
       * Return the size of the real-valued vector representation of
       * the objects current state.
       *
       * @return
       */
      virtual size_t size(void) const = 0;

      /**
       * Return a valarray<double> representing the current object
       * state suitable for use as input to a neural network.
       * @return
       */
      virtual const std::valarray<double>& value(void) const = 0;

      /**
       * Return a ValarrMap representing the current object
       * state suitable for use as input to a neural network.
       *
       * @return
       */
      virtual const ValarrMap& value_map(void) const = 0;

      /**
       * Set the object state from the ValarrMap.
       *
       * @param _vmap
       * @return
       */
      virtual void set(const ValarrMap& _vmap) = 0;

      /**
       * Set the object state from the valarray<double>.
       *
       * @param _vec
       * @return
       */
      virtual void set(const std::valarray<double>& _vec) {};
   };
}

#endif //FLEX_NEURALNET_STATEVIEW_H_
