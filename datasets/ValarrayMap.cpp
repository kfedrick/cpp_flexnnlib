//
// Created by kfedrick on 3/13/21.
//

#include <iostream>
#include "ValarrayMap.h"

using flexnnet::ValarrayMap;

ValarrayMap::ValarrayMap(void)
{
   stale = true;
}

ValarrayMap::ValarrayMap(const ValarrayMap& _valarrmap)
{
   copy(_valarrmap);
}

ValarrayMap::ValarrayMap(const ValarrMap& _values)
{
   stale = true;
   set(_values);
}

ValarrayMap::~ValarrayMap()
{
}

void ValarrayMap::set(const ValarrMap& _values)
{
   data = _values;
   init_virtual_vector();
}

ValarrayMap& ValarrayMap::operator=(const ValarrayMap& _valarrmap)
{
   return copy(_valarrmap);
}

ValarrayMap& ValarrayMap::operator=(const ValarrMap& _values)
{
   stale = true;
   set(_values);
   return *this;
}

ValarrayMap& ValarrayMap::copy(const ValarrayMap& _valarrmap)
{
   data = _valarrmap.data;
   stale = _valarrmap.stale;
   virtual_vector = _valarrmap.virtual_vector;

   return *this;
}

void
ValarrayMap::init_virtual_vector(void) const
{
   if (!stale)
      return;

   size_t sz = 0;
   for (auto entry : data)
      sz += entry.second.size();

   virtual_vector.resize(sz);
   stale = false;
}

void ValarrayMap::concat_virtual_vector(void) const
{
   if (!stale)
      return;

   size_t vndx = 0;
   for (auto entry : data)
      for (auto i = 0; i < entry.second.size(); i++)
         virtual_vector[vndx++] = entry.second[i];

   stale = false;
}
