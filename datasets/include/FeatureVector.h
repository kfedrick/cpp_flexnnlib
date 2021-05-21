//
// Created by kfedrick on 3/13/21.
//

#ifndef FLEX_NEURALNET_STATEVECTOR_H_
#define FLEX_NEURALNET_STATEVECTOR_H_

#include <map>
#include <string>
#include <valarray>
#include <StateView.h>

// Forward declaration for CartesianCoord
namespace flexnnet { class FeatureVector; }

// Forward declarations for stream operators
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::FeatureVector& _coord);
std::istream& operator>>(std::istream& _istrm, flexnnet::FeatureVector& _coord);

namespace flexnnet
{
   class FeatureVector
      : public StateView
   {
   public:
      FeatureVector();
      FeatureVector(const ValarrMap& _values);
      FeatureVector(const FeatureVector& _valarrmap);

      virtual ~FeatureVector();

      size_t
      size(void) const override;

      std::valarray<double>&
      operator[](const std::string& _key);

      std::valarray<double>&
      at(const std::string& _key);

      const std::valarray<double>&
      at(const std::string& _key) const;

      //void
      //set(const ValarrMap& _values);

      FeatureVector&
      operator=(const FeatureVector& _valarrmap);

      FeatureVector&
      operator=(const ValarrMap& _values);

      virtual const std::valarray<double>&
      value(void) const override;

      virtual const ValarrMap&
      value_map(void) const override;

      virtual void
      set(const ValarrMap& _vmap) override;

      friend std::ostream& ::operator<<(std::ostream& _ostrm, const FeatureVector& _valarr);

      friend std::istream& ::operator>>(std::istream& _istrm, FeatureVector& _valarr);

   private:
      FeatureVector&
      copy(const FeatureVector& _datum);

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
   FeatureVector::size() const
   {
      init_virtual_vector();
      return virtual_vector.size();
   }

   inline
   std::valarray<double>&
   FeatureVector::operator[](const std::string& _key)
   {
      stale = true;
      return data[_key];
   }

   inline
   std::valarray<double>&
   FeatureVector::at(const std::string& _key)
   {
      stale = true;
      return data.at(_key);
   }

   inline
   const std::valarray<double>&
   FeatureVector::at(const std::string& _key) const
   {
      return data.at(_key);
   }

   inline
   const std::valarray<double>&
   FeatureVector::value(void) const
   {
      concat_virtual_vector();
      return virtual_vector;
   }

   inline
   const ValarrMap&
   FeatureVector::value_map(void) const
   {
      return data;
   }
}

#endif //FLEX_NEURALNET_STATEVECTOR_H_
