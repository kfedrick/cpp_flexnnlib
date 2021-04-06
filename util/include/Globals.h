//
// Created by kfedrick on 3/12/21.
//

#ifndef FLEX_NEURALNET_GLOBALS_H_
#define FLEX_NEURALNET_GLOBALS_H_

#include <string>
#include <typeinfo>
#include <random>

namespace flexnnet
{
   /**
    * Demangle type id, _name, returned by 'typeid' command.
    * @param name
    * @return
    */
   std::string demangle(const std::string& _name);

   /**
    * Return demangled type id for template type _Typ.
    *
    * @tparam _Typ
    * @return
    */
   template <typename _Typ>
   std::string type_id(void)
   {
      return demangle(typeid(_Typ).name());
   }

   template<typename T>
   class Array2D;

   template <typename T>
   flexnnet::Array2D<T> random_2darray(unsigned int _rows, unsigned int _cols)
   {
      Array2D<T> arr(_rows,_cols);

      std::mt19937_64 rand_engine;

      std::random_device r;
      std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
      rand_engine.seed(seed2);

      std::normal_distribution<T> normal_dist(0.0, 1e-5);
      for (unsigned int row = 0; row < _rows; row++)
         for (unsigned int col = 0; col < _cols; col++)
            arr.at(row,col) = normal_dist(rand_engine);

      return arr;
   }
}
#endif //FLEX_NEURALNET_GLOBALS_H_
