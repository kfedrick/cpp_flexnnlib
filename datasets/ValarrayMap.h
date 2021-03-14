//
// Created by kfedrick on 3/13/21.
//

#ifndef FLEX_NEURALNET_VALARRAYMAP_H_
#define FLEX_NEURALNET_VALARRAYMAP_H_

#include <map>
#include <string>
#include <valarray>
#include <NNetIOValue.h>

namespace flexnnet
{
   class ValarrayMap : public NNetIOValue, public std::map<std::string, std::valarray<double>>
   {
   public:

      virtual const ValarrMap& value_map(void) const override;

      virtual void parse(const ValarrMap& _vmap) override;

   };

   inline
   const ValarrMap& ValarrayMap::value_map(void) const
   {
      return *this;
   }

   inline
   void ValarrayMap::parse(const ValarrMap& _vmap)
   {
   }
}

#endif //FLEX_NEURALNET_VALARRAYMAP_H_
