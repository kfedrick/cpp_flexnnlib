//
// Created by kfedrick on 3/13/21.
//

#include <iostream>
#include "FeatureVector.h"

using flexnnet::FeatureVector;

FeatureVector::FeatureVector(void)
{
   stale = true;
}

FeatureVector::FeatureVector(const FeatureVector& _valarrmap)
{
   copy(_valarrmap);
}

FeatureVector::FeatureVector(const ValarrMap& _values)
{
   stale = true;
   set(_values);
}

FeatureVector::~FeatureVector()
{
}

void FeatureVector::set(const ValarrMap& _values)
{
   data = _values;
   init_virtual_vector();
}

FeatureVector& FeatureVector::operator=(const FeatureVector& _valarrmap)
{
   return copy(_valarrmap);
}

FeatureVector& FeatureVector::operator=(const ValarrMap& _values)
{
   stale = true;
   set(_values);
   return *this;
}

FeatureVector& FeatureVector::copy(const FeatureVector& _valarrmap)
{
   data = _valarrmap.data;
   stale = _valarrmap.stale;
   virtual_vector = _valarrmap.virtual_vector;

   return *this;
}

void
FeatureVector::init_virtual_vector(void) const
{
   if (!stale)
      return;

   size_t sz = 0;
   for (auto entry : data)
      sz += entry.second.size();

   virtual_vector.resize(sz);
   stale = false;
}

void FeatureVector::concat_virtual_vector(void) const
{
   //if (!stale)
   //   return;

   init_virtual_vector();

   size_t vndx = 0;
   for (auto& entry : data)
      for (auto i = 0; i < entry.second.size(); i++)
         virtual_vector[vndx++] = entry.second[i];

   stale = false;
}
