//
// Created by kfedrick on 3/13/21.
//

#ifndef FLEX_NEURALNET_VALARRAYMAP_H_
#define FLEX_NEURALNET_VALARRAYMAP_H_

#include <map>
#include <string>
#include <valarray>
#include <NNetIOInterface.h>

// Forward declaration for CartesianCoord
namespace flexnnet { class ValarrayMap; }

// Forward declarations for stream operators
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::ValarrayMap& _coord);
std::istream& operator>>(std::istream& _istrm, flexnnet::ValarrayMap& _coord);

namespace flexnnet
{
   class ValarrayMap
      : public NNetIOInterface
   {
   public:
      ValarrayMap();
      ValarrayMap(const ValarrMap& _values);
      ValarrayMap(const ValarrayMap& _valarrmap);

      virtual ~ValarrayMap();

      size_t
      size(void) const override;

      std::valarray<double>&
      operator[](const std::string& _key);

      std::valarray<double>&
      at(const std::string& _key);

      const std::valarray<double>&
      at(const std::string& _key) const;

      void
      set(const ValarrMap& _values);

      ValarrayMap&
      operator=(const ValarrayMap& _valarrmap);

      ValarrayMap&
      operator=(const ValarrMap& _values);

      virtual const std::valarray<double>&
      value(void) const override;

      virtual const ValarrMap&
      value_map(void) const override;

      virtual void
      parse(const ValarrMap& _vmap) override;

      friend std::ostream& ::operator<<(std::ostream& _ostrm, const ValarrayMap& _valarr);

      friend std::istream& ::operator>>(std::istream& _istrm, ValarrayMap& _valarr);

   private:
      ValarrayMap&
      copy(const ValarrayMap& _datum);

      void
      init_virtual_vector(void) const;

      void
      concat_virtual_vector(void) const;

   private:
      mutable bool stale;
      mutable std::valarray<double> virtual_vector;
      std::map<std::string, std::valarray<double>> data;
   };

   inline
   size_t
   ValarrayMap::size() const
   {
      return virtual_vector.size();
   }

   inline
   std::valarray<double>&
   ValarrayMap::operator[](const std::string& _key)
   {
      return data[_key];
   }

   inline
   std::valarray<double>&
   ValarrayMap::at(const std::string& _key)
   {
      return data.at(_key);
   }

   inline
   const std::valarray<double>&
   ValarrayMap::at(const std::string& _key) const
   {
      return data.at(_key);
   }

   inline
   const std::valarray<double>&
   ValarrayMap::value(void) const
   {
      concat_virtual_vector();
      return virtual_vector;
   }

   inline
   const ValarrMap&
   ValarrayMap::value_map(void) const
   {
      return data;
   }

   inline
   void
   ValarrayMap::parse(const ValarrMap& _vmap)
   {
      set(_vmap);
   }
}

#endif //FLEX_NEURALNET_VALARRAYMAP_H_
