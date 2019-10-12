//
// Created by kfedrick on 9/27/19.
//

#ifndef FLEX_NEURALNET_VECTORIZABLE_H_
#define FLEX_NEURALNET_VECTORIZABLE_H_

#include <valarray>
#include <memory>
#include "NamedObject.h"

namespace flexnnet
{
   /**
    * Vectorizable defines an interface for classes that may encode their
    * state representation as a real valued vector.
    */
   class Vectorizable : public NamedObject
   {
   public:
      Vectorizable();
      Vectorizable(const std::string& _name);
      Vectorizable(const Vectorizable& _v);

      ~Vectorizable();


   public:
      virtual const std::valarray<double>& vectorize(void) const = 0;
   };

   // Define shared pointer for Vectorizable
   using Vectorizable_sptr_ = std::shared_ptr<Vectorizable>;

   /* -----------------------------------------------
    *    Out-of-line function definitions
    */

   inline Vectorizable::Vectorizable() : NamedObject("Vectorizable")
   {
   }

   inline Vectorizable::Vectorizable(const std::string& _name) : NamedObject(_name)
   {
   }

   inline Vectorizable::Vectorizable(const Vectorizable& _v) : NamedObject(_v.name())
   {
   }

   inline Vectorizable::~Vectorizable()
   {

   }

}

#endif //FLEX_NEURALNET_VECTORIZABLE_H_
