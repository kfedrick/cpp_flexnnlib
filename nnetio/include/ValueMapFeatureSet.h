//
// Created by kfedrick on 5/22/21.
//

#ifndef FLEX_NEURALNET_VALUEMAPFEATURESET_H_
#define FLEX_NEURALNET_VALUEMAPFEATURESET_H_

#include <flexnnet.h>

namespace flexnnet
{
   template<typename Fs> class ValueMapFeatureSet : public Fs
   {
   public:
      ValueMapFeatureSet();
      const ValarrMap& value_map() const;

   protected:
      void gather() const;

   private:
      mutable ValarrMap vmap;
   };

   template<typename Fs>
   inline
   ValueMapFeatureSet<Fs>::ValueMapFeatureSet() : Fs()
   {
   }

   template<typename Fs>
   inline
   void ValueMapFeatureSet<Fs>::gather() const
   {
      size_t sz = Fs::size();
      const std::array<std::string,Fs::SIZE>& f_names = Fs::get_feature_names();
      const std::vector<Feature*>& f_ptrs = Fs::get_feature_pointers();
      for (int ndx = 0; ndx < sz; ndx++)
         vmap[f_names[ndx]] = f_ptrs[ndx]->get_encoding();
   }

   template<typename Fs>
   inline
   const ValarrMap& ValueMapFeatureSet<Fs>::value_map() const
   {
      gather();
      return vmap;
   }
}

#endif // FLEX_NEURALNET_VALUEMAPFEATURESET_H_
