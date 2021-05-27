//
// Created by kfedrick on 10/12/19.
//

#ifndef FLEX_NEURALNET_VECTORIZABLEMAP_H_
#define FLEX_NEURALNET_VECTORIZABLEMAP_H_

#include <map>
#include <memory>
#include <iostream>
#include <sstream>
#include <set>

#include "VectorMap.h"
#include "Vectorizable.h"

namespace flexnnet
{
   template<typename... _Types>
   class VectorizableMap : public VectorMap
   {
   public:
      // --- Constructors, configurators
      //
      VectorizableMap();
      VectorizableMap(_Types& ... vals);

      // --- Member functions from VectorMap interface
      //
      void assign(std::string _name, const std::valarray<double>& _val);

      const std::vector<std::string>& keyset(void) const;

      const std::valarray<double>& operator[](const std::string& _name) const;

      const std::valarray<double>& at(const std::string& _name) const;

      const std::valarray<double>& operator()(void) const;

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
      void assign_impl(T&& _t, Ts&& ... _vals);

      /**
       * Recursive terminal for unpacking variadic constructor arguments.
       * @tparam I
       * @tparam Ts
       */
      template<size_t I, class... Ts>
      void assign_impl(void)
      {}

   private:
      // The container for the Vectorizable objects in this class
      std::map<const std::string, std::shared_ptr<Vectorizable>> data_map;

      // The keyset containing the names of the vectorizable objects in this class
      std::set<std::string> component_names;

      mutable bool stale = true;
      mutable std::valarray<double> virtual_vector;
   };

   template<typename... _Types>
   VectorizableMap<_Types...>::VectorizableMap()
   {
      std::cout << "VectorizableMap()\n";
   }

   template<typename... _Types>
   VectorizableMap<_Types...>::VectorizableMap(_Types& ... _vals)
   {
      std::cout << "VectorizableMap(const _Types&...)\n";
      assign_impl<0>(std::forward<_Types>(_vals)...);
   }

   template<typename... _Types>
   template<size_t I, class T, class... Ts>
   void VectorizableMap<_Types...>::assign_impl(T&& _val, Ts&& ... _vals)
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
         throw std::invalid_argument(err_str.str());
      }

      // Create a new Object using copy constructor, pack into a shared pointer
      // and add it to the data map.
      Vectorizable_sptr_ vptr = Vectorizable_sptr_(new T(std::forward<T>(_val)));
      data_map.insert({name, vptr});
      component_names.insert(name);

      std::cout << "--- " << name << " = " << data_map[name]->vectorize().size() << " "
                << data_map[name]->vectorize()[1] << "\n";

      // Recursively call unpacker
      assign_impl<(I + 1)>(std::forward<Ts>(_vals)...);
   }

   template<typename... _Types>
   const std::vector<std::string>& VectorizableMap<_Types...>::keyset(void) const
   {
      return component_names;
   }

   template<typename... _Types>
   const std::valarray<double>& VectorizableMap<_Types...>::at(const std::string& _key) const
   {
      return data_map.at(_key)->vectorize();
   }

   template<typename... _Types>
   const std::valarray<double>& VectorizableMap<_Types...>::operator[](const std::string& _key) const
   {
      return data_map.at(_key)->vectorize();
   }

   template<typename... _Types>
   const std::valarray<double>& VectorizableMap<_Types...>::operator()(void) const
   {
      if (!stale)
         virtual_vector;

      size_t vndx = 0;
      for (auto& item : data_map)
      {
         const std::valarray<double>& va = item.second->vectorize();
         for (auto i = 0; i < va.size(); i++)
            virtual_vector[vndx++] = va[i];
      }

      stale = false;
      return virtual_vector;
   }

   template<typename... _Types>
   void VectorizableMap<_Types...>::assign(std::string _name, const std::valarray<double>& _val)
   {
      data_map[_name]->assign(_val);
   }

}

#endif //FLEX_NEURALNET_VECTORIZABLEMAP_H_
