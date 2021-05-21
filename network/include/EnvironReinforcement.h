//
// Created by kfedrick on 5/20/21.
//

#ifndef FLEX_NEURALNET_ENVIRONREINFORCEMENT_H_
#define FLEX_NEURALNET_ENVIRONREINFORCEMENT_H_

#include <NetworkInput.h>
#include "Reinforcement.h"

namespace flexnnet
{
   template<unsigned int N>
   class EnvironReinforcement : public NetworkInput, public Reinforcement
   {
   public:
      EnvironReinforcement();
      EnvironReinforcement(const std::vector<std::string>& _fields);
      EnvironReinforcement(const EnvironReinforcement<N>& _reinf);

      virtual EnvironReinforcement<N>& operator=(const EnvironReinforcement<N>& _reinf);

      virtual size_t size() const;

      virtual void fill(double _val);
      virtual double& operator[](size_t _ndx);
      virtual const double& operator[](size_t _ndx) const;
      virtual const double& at(size_t _ndx) const;
      virtual const double& at(const std::string& _field) const;

      virtual const std::vector<std::string>& get_fields() const;
      virtual const std::valarray<double>& value() const;

      /* ******************************************************************
       * NetworkInput interface methods
       */
      virtual const ValarrMap& value_map() const;

   protected:
      virtual void copy(const EnvironReinforcement<N>& _r);

   private:
      std::vector<std::string> fields;
      std::valarray<double> rvalues;
      ValarrMap vmap;
      std::map<std::string, size_t> field_indices_map;
   };

   template<unsigned int N>
   inline
   EnvironReinforcement<N>::EnvironReinforcement()
   {
      fields.resize(N);
      rvalues.resize(N);

      // Initialize value map values to valarrays of size 1.
      std::valarray<double> initial_varr(1);
      std::stringstream field;
      for (int ndx=0; ndx < N; ndx++)
      {
         field.str(std::string());
         field << "R" << ndx;

         fields[ndx] = field.str();
         vmap[field.str()] = initial_varr;
      }
   }

   template<unsigned int N>
   inline
   EnvironReinforcement<N>::EnvironReinforcement(const std::vector<std::string>& _fields)
   {
      // Validate that we provided the specified number of fields, N.
      if (_fields.size() != N)
      {
         static std::stringstream sout;
         sout << "Error : EnvironReinforcement::EnvironReinforcement(_fields) - "
              << "# fields (" << _fields.size() << " != expected size<" << N << ").\n";
         throw std::invalid_argument(sout.str());
      }

      fields.resize(N);
      rvalues.resize(N);

      // Initialize value map values to valarrays of size 1.
      std::valarray<double> initial_varr(1);
      size_t ndx = 0;
      for (auto a_field : _fields)
      {
         fields[ndx++] = a_field;
         vmap[a_field] = initial_varr;
      }
   }

   template<unsigned int N>
   inline
   EnvironReinforcement<N>::EnvironReinforcement(const EnvironReinforcement<N>& _reinf)
   {
      copy(_reinf);
   }

   template<unsigned int N>
   inline
   void EnvironReinforcement<N>::copy(const EnvironReinforcement<N>& _reinf)
   {
      fields = _reinf.fields;
      rvalues = _reinf.rvalues;
      field_indices_map = _reinf.field_indices_map;
   }

   template<unsigned int N>
   inline
   EnvironReinforcement<N>& EnvironReinforcement<N>::operator=(const EnvironReinforcement& _reinf)
   {
      copy(_reinf);
      return *this;
   }

   template<unsigned int N>
   inline
   size_t EnvironReinforcement<N>::size() const
   {
      return rvalues.size();
   }

   template<unsigned int N>
   inline
   const std::vector<std::string>& EnvironReinforcement<N>::get_fields() const
   {
      return fields;
   }

   template<unsigned int N>
   inline
   const std::valarray<double>& EnvironReinforcement<N>::value() const
   {
      return rvalues;
   }

   template<unsigned int N>
   inline
   void EnvironReinforcement<N>::fill(double _val)
   {
      rvalues = _val;
   }

   template<unsigned int N>
   inline
   double& EnvironReinforcement<N>::operator[](size_t _ndx)
   {
      return rvalues[_ndx];
   }

   template<unsigned int N>
   inline
   const double& EnvironReinforcement<N>::operator[](size_t _ndx) const
   {
      return rvalues[_ndx];
   }

   template<unsigned int N>
   inline
   const double& EnvironReinforcement<N>::at(size_t _ndx) const
   {
      return rvalues[_ndx];
   }

   template<unsigned int N>
   inline
   const double& EnvironReinforcement<N>::at(const std::string& _field) const
   {
      return rvalues[field_indices_map.at(_field)];
   }

   template<unsigned int N>
   inline
   const ValarrMap& EnvironReinforcement<N>::value_map() const
   {
      return vmap;
   }
}

#endif //FLEX_NEURALNET_ENVIRONREINFORCEMENT_H_
