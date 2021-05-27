//
// Created by kfedrick on 5/22/21.
//

#ifndef FLEX_NEURALNET_FEATURE_H_
#define FLEX_NEURALNET_FEATURE_H_

#include <string>
#include <iostream>

namespace flexnnet
{
   class Feature
   {
   protected:
      Feature();
      Feature(size_t _sz=1);
      Feature(const Feature& _f);

   public:
      Feature& operator=(const Feature& _f);

      virtual size_t size() const;
      virtual const std::valarray<double>& get_encoding() const;
      virtual void decode(const std::valarray<double>& _encoding);

   protected:
      void copy(const Feature& _f);

   protected:
      const std::valarray<double>& const_encoding_ref;

   private:
      std::valarray<double> encoding;
   };

   inline
   Feature::Feature() : const_encoding_ref(encoding) {}

   inline
   Feature::Feature(size_t _sz) : const_encoding_ref(encoding), encoding(_sz)
   {
   }

   inline
   Feature::Feature(const Feature& _f) : const_encoding_ref(encoding)
   {
      copy(_f);
   }

   inline
   Feature& Feature::operator=(const Feature& _f)
   {
      copy(_f);
   }

   inline
   void Feature::copy(const Feature& _f)
   {
      encoding = _f.encoding;
   }

   inline
   size_t Feature::size() const
   {
      return encoding.size();
   }

   inline
   const std::valarray<double>& Feature::get_encoding() const
   {
      return encoding;
   }

   inline
   void Feature::decode(const std::valarray<double>& _encoding)
   {
      encoding = _encoding;
   }
}

#endif // FLEX_NEURALNET_FEATURE_H_
