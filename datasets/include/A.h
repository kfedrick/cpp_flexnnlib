//
// Created by kfedrick on 9/28/19.
//

#ifndef FLEX_NEURALNET_A_H_
#define FLEX_NEURALNET_A_H_

#include "Vectorizable.h"

namespace flexnnet
{
   class A : public Vectorizable
   {
   public:
      A();
      A(const std::string _name, const std::valarray<double>&& _vec);
      A(const A& _a);
      A(const A&& _a);
      virtual size_t size(void) const;
      virtual const std::valarray<double>& vectorize(void) const;

   private:
      std::string vname;
      std::valarray<double> data;
   };

   inline A::A() : Vectorizable()
   {
      std::cout << "A()\n";
   }

   inline A::A(const A& _a) : Vectorizable(_a)
   {
      std::cout << "A(const A&)\n";
      data = _a.data;
   }

   inline A::A(const A&& _a) : Vectorizable(_a)
   {
      std::cout << "A(const A&&)\n";
      data = std::move(_a.data);
   }

   inline A::A(const std::string _name, const std::valarray<double>&& _vec) : Vectorizable(_name)
   {
      std::cout << "A(const std::string, const std::valarray<double)\n";
      data = _vec;
   }

   inline size_t A::size(void) const
   {
      return data.size();
   }

   inline const std::valarray<double>& A::vectorize(void) const
   {
      return data;
   }
}

#endif //_A_H_
