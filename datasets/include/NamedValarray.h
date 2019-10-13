//
// Created by kfedrick on 10/12/19.
//

#ifndef FLEX_NEURALNET_NAMEDVALARRAY_H_
#define FLEX_NEURALNET_NAMEDVALARRAY_H_

#include <valarray>
#include <util/include/NamedObject.h>
#include <sstream>
#include "Vectorizable.h"

namespace flexnnet
{
   template<size_t _SZ>
   class NamedValarray : public NamedObject, public Vectorizable
   {
   public:
      NamedValarray();
      NamedValarray(const std::string& _name);
      NamedValarray(const NamedValarray& _valarr);

      // --- Vectorizable interface methods
      //

      virtual const std::valarray<double>& vectorize(void) const;

      virtual const Vectorizable& decode(const std::valarray<double>& _data);

      virtual const Vectorizable& decode(const std::valarray<double>&& _data);

   private:
      std::valarray<double> data;
   };

   template<size_t _SZ>
   NamedValarray<_SZ>::NamedValarray() : NamedObject("Valarray")
   {
      data.resize(_SZ);
   }

   template<size_t _SZ>
   NamedValarray<_SZ>::NamedValarray(const std::string& _name) : NamedObject(_name)
   {
      data.resize(_SZ);
   }

   template<size_t _SZ>
   NamedValarray<_SZ>::NamedValarray(const NamedValarray<_SZ>& _valarr) : NamedObject(_valarr.name())
   {
      data.resize(_valarr.data.size());
      data = _valarr.data;
   }

   template<size_t _SZ>
   const std::valarray<double>& NamedValarray<_SZ>::vectorize(void) const
   {
      return data;
   }

   template<size_t _SZ>
   const Vectorizable& NamedValarray<_SZ>::decode(const std::valarray<double>& _data)
   {
      // If key is doesn't exist, throw exception
      if (_SZ == _data.size())
      {
         std::ostringstream err_str;
         err_str
            << "Error : decode(std::valarray<double>) - Incorrect size ("
            << _data.size() << ") for valarray argument - expected (" << _SZ << ").\n";
         throw std::invalid_argument(err_str.str());
      }

      data = _data;
      return *this;
   }

   template<size_t _SZ>
   const Vectorizable& NamedValarray<_SZ>::decode(const std::valarray<double>&& _data)
   {
      // If key is doesn't exist, throw exception
      if (_SZ == _data.size())
      {
         std::ostringstream err_str;
         err_str
            << "Error : decode(std::valarray<double>) - Incorrect size ("
            << _data.size() << ") for valarray argument - expected (" << _SZ << ").\n";
         throw std::invalid_argument(err_str.str());
      }

      data = std::move(_data);
      return *this;
   }
}

#endif //FLEX_NEURALNET_NAMEDVALARRAY_H_
