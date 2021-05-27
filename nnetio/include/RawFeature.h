//
// Created by kfedrick on 5/23/21.
//

#ifndef FLEX_NEURALNET_RAWFEATURE_H_
#define FLEX_NEURALNET_RAWFEATURE_H_

#include <FixedSizeFeature.h>

namespace flexnnet { template<size_t N> class RawFeature; }

// Forward declarations for stream operators
template<size_t N>
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::RawFeature<N>& _feature);

template<size_t N>
std::istream& operator>>(std::istream& _istrm, flexnnet::RawFeature<N>& _feature);

namespace flexnnet
{
   template<size_t N>
   class RawFeature : public FixedSizeFeature<N>
   {
   public:
      RawFeature() : FixedSizeFeature<N>() {};
      //RawFeature& operator=(const RawFeature& _f);

      const std::valarray<double>& value() const;
      //std::valarray<double>& value();
   };

/*   template<size_t N>
   RawFeature<N>& RawFeature<N>::operator=(const RawFeature& _f)
   {

      return *this;
   }*/

/*   template<size_t N>
   std::valarray<double>& RawFeature<N>::value()
   {
      return Feature::encoding;
   }*/

   template<size_t N>
   const std::valarray<double>& RawFeature<N>::value() const
   {
      return Feature::const_encoding_ref;
   }
}

#endif // FLEX_NEURALNET_RAWFEATURE_H_
