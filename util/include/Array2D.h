/*
 * Array2.h
 *
 *  Created on: Jan 30, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ARRAY2D_H_
#define FLEX_NEURALNET_ARRAY2D_H_

#include <vector>
#include <valarray>
#include <sstream>
#include <stdexcept>

namespace flexnnet
{
   template<class T>
   class Array2D
   {
   public:
      struct Dimensions
      {
         size_t rows;
         size_t cols;
      };

      typedef Array2D<T>& type_ref;
      typedef const Array2D<T>& const_type_ref;

      using value_type = T;
      using difference_type = T;
      using value_ptr = T*;
      using value_ref = T&;
      using const_value_ref = const T&;
      using size_ref = Dimensions&;
      using const_size_ref = const Dimensions&;

   public:

      /* *****************************************
       *   Constructors and configuration
       * *****************************************
       */
      Array2D();
      Array2D(size_t _rows, size_t _cols);
      Array2D(const std::vector<std::vector<T> >& _arr);
      Array2D(const Array2D<T>& _arr);

      void resize(size_t _rows, size_t _cols, T _val = T());

      /* *****************************************
       *   Getters/setters
       * *****************************************
       */
      const_size_ref& size() const;
      static Dimensions size(const std::vector<std::vector<T>>& _arr);

      value_ref at(size_t _rowndx, size_t _colndx);
      const_value_ref at(size_t _rowndx, size_t) const;
      const_value_ref operator()(size_t _rowndx, size_t _colndx) const;

      /* *****************************************
       *   Utility functions
       * *****************************************
       */
      void set(T _val = T());
      void set(const std::vector<std::vector<T> >& _arr);
      void set(const Array2D<T>& arr);

      type_ref operator=(const_value_ref val);
      type_ref operator=(const_type_ref arr);
      type_ref operator=(const std::vector<std::vector<T> >& _arr);
      type_ref operator+=(const_type_ref _arr);
      type_ref operator-=(const_type_ref arr);
      type_ref operator*=(const_type_ref _arr);

   private:
      static bool validate_dimensions(const_type_ref _arr1, const_type_ref arr2);
      static bool validate_dimensions(const_size_ref _ldim, const_size_ref _rdim);

   private:
      Dimensions dimensions;
      std::valarray<T> data;
   };

   template<class T> inline Array2D<T>::Array2D()
   {
      dimensions = {.rows=0, .cols=0};
   }

   template<class T> inline Array2D<T>::Array2D(const Array2D<T>& _arr)
   {
      set(_arr);
   }

   template<class T> inline Array2D<T>::Array2D(size_t rows, size_t cols)
   {
      resize(rows, cols);
   }

   template<class T> inline Array2D<T>::Array2D(const std::vector<std::vector<T> >& _arr)
   {
      dimensions = {.rows=0, .cols=0};
      set(_arr);
   }

   template<class T> inline
   const typename Array2D<T>::Dimensions& Array2D<T>::size() const
   {
      return dimensions;
   }

   template<class T>
   typename Array2D<T>::Dimensions Array2D<T>::size(const std::vector<std::vector<T>>& _arr)
   {
      typename Array2D<T>::Dimensions dims;

      dims.rows = _arr.size();
      dims.cols = 0;

      // Set number of columns to the size of the first vector
      if (dims.rows > 0)
         dims.cols = _arr[0].size();

      // Validate that the remaining rows have the same number of columns
      for (unsigned int row_ndx = 0; row_ndx < dims.rows; row_ndx++)
      {
         // Validate the size of each row vector
         if (_arr[row_ndx].size() != dims.cols)
         {
            std::ostringstream err_str;
            err_str
               << "Error : Array2D<T>(vector<vector<T>>) - mismatch column size on row "
               << row_ndx << " (" << _arr[row_ndx].size() << ") " << " : expected " << dims.cols << "."
               << std::endl;
            throw std::invalid_argument(err_str.str());
         }
      }
      return dims;
   }

   template<class T> inline T& Array2D<T>::at(size_t _rowndx, size_t _colndx)
   {
      std::ostringstream err_str;

      if (_rowndx > dimensions.rows)
      {
         err_str
            << "Error : Array2<T>::at(rowndx, colndx) - row index " << _rowndx << " out-of-range\n";
         throw std::out_of_range(err_str.str());
      }

      if (_colndx > dimensions.cols)
      {
         err_str
            << "Error : Array2<T>::at(rowndx, colndx) - column index " << _colndx << " out-of-range\n";
         throw std::out_of_range(err_str.str());
      }

      return data[_rowndx + _colndx * dimensions.rows];
   }

   template<class T> inline const T& Array2D<T>::at(size_t _rowndx, size_t _colndx) const
   {
      std::ostringstream err_str;

      if (_rowndx > dimensions.rows)
      {
         err_str
            << "Error : Array2<T>::at(rowndx, colndx) - row index " << _rowndx << " out-of-range\n";
         throw std::out_of_range(err_str.str());
      }

      if (_colndx > dimensions.cols)
      {
         err_str
            << "Error : Array2<T>::at(rowndx, colndx) - column index " << _colndx << " out-of-range\n";
         throw std::out_of_range(err_str.str());
      }

      return data[_rowndx + _colndx * dimensions.rows];
   }

   template<class T> inline const T& Array2D<T>::operator()(size_t _rowndx, size_t _colndx) const
   {
      return data[_rowndx + _colndx * dimensions.rows];
   }

   template<class T> inline
   void Array2D<T>::set(T val)
   {
      data = val;
   }

   template<class T> inline
   void Array2D<T>::set(const Array2D<T>& _arr)
   {
      resize(_arr.size().rows, _arr.size().cols);
      data = _arr.data;
   }

   template<class T> inline
   void Array2D<T>::set(const std::vector<std::vector<T> >& _arr)
   {
      // Get dimensionality of source array
      Dimensions src_dims = Array2D<T>::size(_arr);
      resize(src_dims.rows, src_dims.cols);

      for (unsigned int row_ndx = 0; row_ndx < dimensions.rows; row_ndx++)
         for (unsigned int col_ndx = 0; col_ndx < dimensions.cols; col_ndx++)
            this->at(row_ndx, col_ndx) = _arr[row_ndx][col_ndx];
   }

   template<class T> inline
   void Array2D<T>::resize(size_t _rows, size_t _cols, T _fill)
   {
      if ((_rows == 0 && _cols > 0) || (_rows > 0 && _cols == 0))
      {
         std::ostringstream err_str;
         err_str
            << "Error: Array2<T>::resize(int,int) - invalid array dimension ("
            << _rows << "," << _cols << ")";
         throw std::invalid_argument(err_str.str());
      }

      data.resize(_rows * _cols);
      data = _fill;
      dimensions = {.rows=_rows, .cols=_cols};
   }

   template<class T> inline Array2D<T>& Array2D<T>::operator=(const_value_ref _val)
   {
      set(_val);
      return *this;
   }

   template<class T> inline Array2D<T>& Array2D<T>::operator=(const_type_ref _arr)
   {
      validate_dimensions(*this, _arr);

      dimensions = _arr.dimensions;
      data = _arr.data;
      return *this;
   }

   template<class T> inline Array2D<T>& Array2D<T>::operator=(const std::vector<std::vector<T> >& _arr)
   {
      Dimensions src_dims = Array2D<T>::size(_arr);
      validate_dimensions(this->dimensions, src_dims);

      set(_arr);
      return *this;
   }

   template<class T> inline Array2D<T>& Array2D<T>::operator+=(const_type_ref _arr)
   {
      validate_dimensions(*this, _arr);
      data += _arr.data;

      return *this;
   }

   template<class T> inline Array2D<T>& Array2D<T>::operator-=(const_type_ref _arr)
   {
      validate_dimensions(*this, _arr);
      data -= _arr.data;

      return *this;
   }

   template<class T> inline Array2D<T>& Array2D<T>::operator*=(const_type_ref _arr)
   {
      validate_dimensions(*this, _arr);
      data *= _arr.data;

      return *this;
   }

   template<class T> inline Array2D<T> operator+(const Array2D<T>& _larr,
                                                 const Array2D<T>& _rarr)
   {
      Array2D<T>::validate_dimensions(_larr, _rarr);

      Array2D<T> ret_arr(_larr);
      ret_arr += _rarr;

      return ret_arr;
   }

   template<class T> inline bool Array2D<T>::validate_dimensions(const_type_ref _larr, const_type_ref _rarr)
   {
      const_size_ref& ldim = _larr.size();
      const_size_ref& rdim = _rarr.size();

      if ((ldim.rows != rdim.rows || ldim.cols != rdim.cols))
      {
         std::ostringstream err_str;
         err_str
            << "Error : Array2<T>::operator+(Array2<T>, Array2<T>) - mismatched array dimension :\n"
            << "arg 1 :(" << ldim.rows << "," << ldim.cols << ")\n"
            << "arg 2 :(" << rdim.rows << "," << rdim.cols << ")";
         throw std::invalid_argument(err_str.str());
      }

      return true;
   }

   template<class T> inline bool Array2D<T>::validate_dimensions(const_size_ref _ldim, const_size_ref _rdim)
   {
      if ((_ldim.rows != _rdim.rows || _ldim.cols != _rdim.cols))
      {
         std::ostringstream err_str;
         err_str
            << "Error : Array2<T>::operator+(Array2<T>, Array2<T>) - mismatched array dimension :\n"
            << "arg 1 :(" << _ldim.rows << "," << _ldim.cols << ")\n"
            << "arg 2 :(" << _rdim.rows << "," << _rdim.cols << ")";
         throw std::invalid_argument(err_str.str());
      }

      return true;
   }
} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_ARRAY2D_H_ */

