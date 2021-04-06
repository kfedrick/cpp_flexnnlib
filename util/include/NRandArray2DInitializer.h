//
// Created by kfedrick on 3/28/21.
//

#ifndef FLEX_NEURALNET_NRANDARRAY2DINITIALIZER_H_
#define FLEX_NEURALNET_NRANDARRAY2DINITIALIZER_H_

#include <Array2D.h>
#include <random>

namespace flexnnet
{
   template <typename T>
   class NRandArray2DInitializer
   {
   public:
      NRandArray2DInitializer(T _lower, T _upper);
      virtual ~NRandArray2DInitializer();

      Array2D<T> operator()(const typename Array2D<T>::Dimensions& _dims);

   private:
      T mean;
      T stdev;

      bool init_flag;
      std::mt19937_64 rand_engine;
   };

   template <typename T>
   NRandArray2DInitializer<T>::NRandArray2DInitializer(T _mean, T _stdev)
   {
      mean = _mean;
      stdev = _stdev;

      if (!init_flag)
      {
         std::random_device r;
         std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
         rand_engine.seed(seed2);
      }
   }

   template <typename T>
   NRandArray2DInitializer<T>::~NRandArray2DInitializer() {}

   template <typename T>
   flexnnet::Array2D<T> NRandArray2DInitializer<T>::operator()(const typename Array2D<T>::Dimensions& _dims)
   {
      Array2D<T> arr(_dims.rows,_dims.cols);

      std::normal_distribution<T> normal_dist(mean, stdev);
      for (unsigned int row = 0; row < _dims.rows; row++)
         for (unsigned int col = 0; col < _dims.cols; col++)
            arr.at(row,col) = normal_dist(rand_engine);

      return arr;
   }
}

#endif //FLEX_NEURALNET_NRANDARRAY2DINITIALIZER_H_
