//
// Created by kfedrick on 9/28/19.
//

#ifndef FLEX_NEURALNET_A_H_
#define FLEX_NEURALNET_A_H_

#include "Vectorizable.h"

namespace flexnnet
{
   class A : public NamedObject, public Vectorizable
   {
   public:
      A();
      A(const std::string _name, const std::valarray<double>&& _vec);
      A(const A& _a);
      A(const A&& _a);
      virtual size_t size(void) const;
      virtual const std::valarray<double>& vectorize(void) const;
      virtual const Vectorizable& assign(const std::valarray<double>& _val);

   private:
      std::string vname;
      std::valarray<double> data;
   };

   inline A::A() : NamedObject("A"), Vectorizable()
   {
      std::cout << "A()\n";
   }

   inline A::A(const A& _a) : NamedObject(_a.name()), Vectorizable(_a)
   {
      std::cout << "A(const A&)\n";
      data = _a.data;
   }

   inline A::A(const A&& _a) : NamedObject(_a.name()), Vectorizable(_a)
   {
      std::cout << "A(const A&&)\n";
      data = std::move(_a.data);
   }

   inline A::A(const std::string _name, const std::valarray<double>&& _vec) : NamedObject(_name), Vectorizable()
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

   inline const Vectorizable& A::assign(const std::valarray<double>& _val)
   {
      return *this;
   }
}

#endif //_A_H_
