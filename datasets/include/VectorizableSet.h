//
// Created by kfedrick on 9/28/19.
//

#ifndef FLEX_NEURALNET_VECTORIZABLESET_H_
#define FLEX_NEURALNET_VECTORIZABLESET_H_

#include <iostream>
#include <valarray>
#include <map>
#include <memory>

#include "Vectorizable.h"

namespace flexnnet
{
   template<typename... _Types>
   class VectorizableSet
   {
   protected:
      // The number of Vectorizable object in this class
      static const int size = sizeof...(_Types);

      // Type specification for shared pointer to a Vectorizable object
      using Vectorizable_ptr_ = std::shared_ptr<Vectorizable>;

   public:
      VectorizableSet(_Types&... vals);
      VectorizableSet(_Types&&... vals);

      /**
       * Return the vectorization of the object named by key.
       * @param _key
       * @return
       */
      const std::valarray<double>& at(const std::string& _key) const;

      /**
       * Return a vector with the coelesced values of all vectorizable objects
       * in this object.
       *
       * @return
       */
      const std::valarray<double>& concat(void) const;

      
   private:

      /**
       * Recursive template class to unpack variadic constructor arguments
       * and save them to local object members.
       *
       * @tparam I  - Index parameter
       * @tparam T  - Next object type
       * @tparam Ts - Remaining object types
       * @param _t  - Next vectorizable object reference
       * @param _ts - Remaining vectorizable object references
       */
      template<size_t I, class T, class... Ts>
      void assign_impl (T&& _t, Ts&& ... _vals);

      /**
       * Recursive terminal for unpacking variadic constructor arguments.
       * @tparam I
       * @tparam Ts
       */
      template<size_t I, class... Ts>
      void assign_impl (void)
      {}
      
   private:
      // The container for the Vectorizable objects in this class
      std::map<const std::string, std::shared_ptr<const Vectorizable>> data_map;

      mutable bool stale = true;
      mutable std::valarray<double> virtual_vector;
   };

   template<typename... _Types>
   VectorizableSet< _Types...>::VectorizableSet(_Types&... _vals)
   {
      std::cout << "VectorizableSet(const _Types&...)\n";
      assign_impl<0> (std::forward<_Types> (_vals)...);
   }

   template<typename... _Types>
   VectorizableSet< _Types...>::VectorizableSet(_Types&&... _vals)
   {
      std::cout << "VectorizableSet(const _Types&&...)\n";
      assign_impl<0> (std::forward<_Types> (_vals)...);
   }

   template<typename... _Types>
   template<size_t I, class T, class... Ts>
   void VectorizableSet< _Types...>::assign_impl (T&& _val, Ts&& ... _vals)
   {
      std::cout << "assign_impl(T&&, Ts&&)\n";
      std::cout << _val.name().c_str() << " " << I << " " << sizeof...(Ts) << "\n";

      // Get the object name
      std::string name = _val.name();

      // If the object name is already in the data map throw an exception.
      if (data_map.find(name) != data_map.end())
      {
         std::ostringstream err_str;
         err_str
            << "Error : VectorizableSet() - key '" << name << "' already exists.\n";
         throw std::invalid_argument (err_str.str ());
      }

      // Create a new Object using copy constructor, pack into a shared pointer
      // and add it to the data map.
      Vectorizable_sptr_ vptr = Vectorizable_ptr_(new T(std::forward<T>(_val)));
      data_map.insert( { _val.name(), vptr } );

      std::cout << "--- " << name << " = " << data_map[name]->vectorize().size() << " " <<  data_map[name]->vectorize()[1] << "\n";

      // Recursively call unpacker
      assign_impl<(I + 1)> (std::forward<Ts> (_vals)...);
   }


   template<typename... _Types>
   const std::valarray<double>& VectorizableSet< _Types...>::at(const std::string& _key) const
   {
      return data_map.at(_key)->vectorize();
   }

   template<typename... _Types>
   const std::valarray<double>& VectorizableSet< _Types...>::concat(void) const
   {
      if (!stale)
         virtual_vector;

      size_t vndx = 0;
      for (auto& item : data_map)
      {
         const std::valarray<double>& va = item.second->vectorize ();
         for (auto i = 0; i < va.size (); i++)
            virtual_vector[vndx++] = va[i];
      }

      stale = false;
      return virtual_vector;
   }

}

#endif //FLEX_NEURALNET_VECTORIZABLESET_H_
