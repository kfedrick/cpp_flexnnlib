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
    * Vectorizable defines an interface for classes that may activate their
    * internal state representation as a real valued vector.
    */
   class Vectorizable
   {
   public:
      Vectorizable();
      Vectorizable(std::valarray<double>& _valarr);

      virtual const std::valarray<double>& vectorize(void) const;

      virtual const Vectorizable& assign(const std::valarray<double>& _vdata);

   private:
      std::valarray<double>* vdata_ptr = NULL;
   };

   inline
   Vectorizable::Vectorizable()
   {
   }


   inline
   Vectorizable::Vectorizable(std::valarray<double>& _valarr)
   {
      vdata_ptr = &_valarr;
   }

   const std::valarray<double>& Vectorizable::vectorize(void) const
   {
      return *vdata_ptr;
   };

   const Vectorizable& Vectorizable::assign(const std::valarray<double>& _vdata)
   {
      *vdata_ptr = _vdata;
   };

   // Define shared pointer for Vectorizable
   typedef std::shared_ptr<Vectorizable> Vectorizable_sptr_;
}

#endif //FLEX_NEURALNET_VECTORIZABLE_H_
