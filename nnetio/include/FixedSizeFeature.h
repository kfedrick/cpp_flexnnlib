//
// Created by kfedrick on 5/22/21.
//

#ifndef FLEX_NEURALNET_FIXEDSIZEFEATURE_H_
#define FLEX_NEURALNET_FIXEDSIZEFEATURE_H_

#include <Feature.h>


namespace flexnnet
{
   template<size_t N>
   class FixedSizeFeature : public Feature
   {
   protected:
      FixedSizeFeature();

   public:
      virtual void decode(const std::valarray<double>& _encoding) override;
   };

   template<size_t N>
   inline
   FixedSizeFeature<N>::FixedSizeFeature() : Feature(N)
   {
   }

   template<size_t N>
   inline
   void FixedSizeFeature<N>::decode(const std::valarray<double>& _encoding)
   {
      // TODO - validate size of valarray
      Feature::decode(_encoding);
   }
}

#endif // FLEX_NEURALNET_FIXEDSIZEFEATURE_H_
