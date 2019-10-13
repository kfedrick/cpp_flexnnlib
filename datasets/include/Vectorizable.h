//
// Created by kfedrick on 9/27/19.
//

#ifndef FLEX_NEURALNET_VECTORIZABLE_H_
#define FLEX_NEURALNET_VECTORIZABLE_H_

#include <valarray>
#include <memory>

namespace flexnnet
{
   /**
    * Vectorizable defines an interface for classes that may encode their
    * internal state representation as a real valued vector.
    */
   class Vectorizable
   {
   public:
      virtual const std::valarray<double>& vectorize(void) const = 0;
      virtual const Vectorizable& assign(const std::valarray<double>& _val) = 0;
   };

   // Define shared pointer for Vectorizable
   typedef std::shared_ptr<Vectorizable> Vectorizable_sptr_;

}

#endif //FLEX_NEURALNET_VECTORIZABLE_H_
