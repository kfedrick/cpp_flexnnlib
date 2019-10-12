//
// Created by kfedrick on 9/25/19.
//

#ifndef FLEX_NEURALNET_OLDVECTORSET_H_
#define FLEX_NEURALNET_OLDVECTORSET_H_

#include <tuple>
#include <valarray>
#include <iostream>
#include <vector>

#include "VectorConcatenator.h"
#include "Vectorizable.h"

namespace flexnnet
{
   template<typename... _Types>
   class OldVectorSet : public VectorConcatenator
   {
      using Vectorizable_ptr_ = std::shared_ptr<const Vectorizable>;

   public:
      static const int size = sizeof...(_Types);

   public:
      OldVectorSet(_Types& ... vals);
      OldVectorSet(_Types&& ... vals);

      const std::valarray<double>& operator[](const std::string& _key) const;

      template<size_t I, class... Ts>
      void doit_impl(const std::tuple<Ts...>&& tuple)
      {
         std::cout << std::get<I>(tuple) << "\n";
         data_map.emplace({std::get<I>(tuple).name(), std::get<I>(tuple)});

         if (I + 1 < sizeof... (Ts))
         {
            doit_impl<(I + 1 < sizeof...(Ts) ? I + 1 : I)>(tuple);
         }
      }

      template<class... Ts>
      void doit(const std::tuple<Ts...>&& tuple)
      {
         doit_impl<0>();
      }

   private:
      //std::tuple<_Types...> my_tuple;
      std::map<const std::string, std::shared_ptr<const Vectorizable>> data_map;
   };

   template<typename... _Types>
   OldVectorSet<_Types...>::OldVectorSet(_Types& ... vals)
   {
      std::cout << "OldVectorSet(const _Types&...)\n";
   }

   template<typename... _Types>
   OldVectorSet<_Types...>::OldVectorSet(_Types&& ... vals)
   {
      std::cout << "OldVectorSet(const _Types&&...)\n";
   }

   template<typename... _Types>
   const std::valarray<double>& OldVectorSet<_Types...>::operator[](const std::string& _key) const
   {
      return data_map.at(_key)->vectorize();
   }

}

#endif //FLEX_NEURALNET_OLDVECTORSET_H_
