/*
 * Array2.h
 *
 *  Created on: Jan 30, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ARRAY_H_
#define FLEX_NEURALNET_ARRAY_H_

#include <vector>
#include <sstream>
#include <stdexcept>
#include "Reinforcement.h"

namespace flexnnet
{

   template<class T>
   class Vector
   {
   public:

      /* *****************************************
       *   Constructors and configuration
       * *****************************************
       */
      Vector(unsigned int sz = 0);
      Vector(const std::vector<T>& vec);
      void resize(unsigned int sz, T& val = T());

      /* *****************************************
       *   Getters/setters
       * *****************************************
       */
      unsigned int size() const;

      T& operator[](unsigned int ndx);
      const T& operator[](unsigned int ndx) const;

      T& at(unsigned int ndx);
      const T& at(unsigned int ndx) const;

      void operator=(const std::vector<T>& vec);
      void operator=(const T& val);
      Vector<T>& operator+=(const std::vector<T>& vec);
      Vector<T>& operator*=(const std::vector<T>& vec);

      operator std::vector<T>();
      operator std::vector<T>&();
      operator std::vector<T>*();

   private:
      std::vector<T> data;
   };

   template<class T> inline Vector<T>::Vector(unsigned int sz)
   {
      data.resize(sz);
   }

   template<class T> inline Vector<T>::Vector(const std::vector<T>& vec)
   {
      data = vec;
   }

   template<class T> inline
   void Vector<T>::resize(unsigned int sz, T& val)
   {
      data.resize(sz, val);
   }

   template<class T> inline
   unsigned int Vector<T>::size() const
   {
      return data.size();
   }

   template<class T> inline T& Vector<T>::operator[](unsigned int ndx)
   {
      return data[ndx];
   }

   template<class T> inline const T& Vector<T>::operator[](unsigned int ndx) const
   {
      return data[ndx];
   }

   template<class T> inline T& Vector<T>::at(unsigned int ndx)
   {
      return data.at(ndx);
   }

   template<class T> inline const T& Vector<T>::at(unsigned int ndx) const
   {
      return data.at(ndx);
   }

   template<class T> inline
   void Vector<T>::operator=(const std::vector<T>& vec)
   {
      data = vec;
   }

   template<class T> inline
   void Vector<T>::operator=(const T& val)
   {
      data.assign(data.size(), val);
   }

   template<class T> inline Vector<T>::operator std::vector<T>()
   {
      return data;
   }

   template<class T> inline Vector<T>::operator std::vector<T>&()
   {
      return data;
   }

   template<class T> inline Vector<T>::operator std::vector<T>*()
   {
      return &data;
   }

   template<class T> inline Vector<T>& Vector<T>::operator+=(const std::vector<T>& vec)
   {
      unsigned int sz = vec.size();
      for (int ndx = 0; ndx < sz; ndx++)
         this->at(ndx) += vec.at(ndx);

      return *this;
   }

   template<class T> inline Vector<T>& Vector<T>::operator*=(const std::vector<T>& vec)
   {
      unsigned int sz = vec.size();
      for (int ndx = 0; ndx < sz; ndx++)
         this->at(ndx) *= vec.at(ndx);

      return *this;
   }

   template<class T> inline Vector<T> operator-(Vector<T>& v1, Vector<T>& v2)
   {
      unsigned int sz = v1.size();

      Vector<T> ret_vec(sz);
      for (unsigned int ndx = 0; ndx < sz; ndx++)
         ret_vec.at(ndx) = v1.at(ndx) - v2.at(ndx);

      return ret_vec;
   }

   template<class T> inline Vector<T> operator*(double val, Vector<T>& vec)
   {
      unsigned int sz = vec.size();

      Vector<T> ret_vec(sz);
      for (unsigned int ndx = 0; ndx < sz; ndx++)
         ret_vec.at(ndx) = val * vec.at(ndx);

      return ret_vec;
   }

   template<class T> inline Vector<T> operator*(Vector<T>& vec, double val)
   {
      unsigned int sz = vec.size();

      Vector<T> ret_vec(sz);
      for (unsigned int ndx = 0; ndx < sz; ndx++)
         ret_vec.at(ndx) = val * vec.at(ndx);

      return ret_vec;
   }

   template<class T>
   class Array
   {
   public:

      /* *****************************************
       *   Constructors and configuration
       * *****************************************
       */
      Array();
      Array(unsigned int rows, unsigned int cols);
      void resize(unsigned int rows, unsigned int columns, T val = T());

      /* *****************************************
       *   Getters/setters
       * *****************************************
       */
      unsigned int rowDim() const;
      unsigned int colDim() const;

      std::vector<T>& operator[](unsigned int rowIdx);
      const std::vector<T>& operator[](unsigned int rowIdx) const;

      T& at(unsigned int rowIdx, unsigned int colIdx);
      const T& at(unsigned int rowIdx, unsigned int colIdx) const;

      /* *****************************************
       *   Utility functions
       * *****************************************
       */
      void set(T val = T());
      void set(const Array<T>& arr);
      void set(const std::vector<std::vector<T> >& arr);

      Array<T>& operator=(const T& val);
      Array<T>& operator=(const Array<T>& arr);
      Array<T>& operator+=(const Array<T>& arr);

   private:

      void copy(const Array<T>& arr);

      unsigned int row_dim;
      unsigned int col_dim;
      std::vector<std::vector<T> > data;
   };

   template<class T> inline Array<T>::Array()
   {
      row_dim = 0;
      col_dim = 0;
   }

   template<class T> inline Array<T>::Array(unsigned int rows, unsigned int cols)
   {
      resize(rows, cols);
   }

   template<class T> inline
   unsigned int Array<T>::rowDim() const
   {
      return row_dim;
   }

   template<class T> inline
   unsigned int Array<T>::colDim() const
   {
      return col_dim;
   }

   template<class T> inline std::vector<T>& Array<T>::operator[](unsigned int rowIdx)
   {
      return data[rowIdx];
   }

   template<class T> inline const std::vector<T>& Array<T>::operator[](
      unsigned int rowIdx) const
   {
      return data[rowIdx];
   }

   template<class T> inline T& Array<T>::at(unsigned int rowIdx,
                                            unsigned int colIdx)
   {
      return data.at(rowIdx).at(colIdx);
   }

   template<class T> inline const T& Array<T>::at(unsigned int rowIdx,
                                                  unsigned int colIdx) const
   {
      return data.at(rowIdx).at(colIdx);
   }

   template<class T> inline
   void Array<T>::set(T val)
   {
      for (unsigned int i = 0; i < row_dim; i++)
         for (unsigned int j = 0; j < col_dim; j++)
            data[i][j] = val;
   }

   template<class T> inline
   void Array<T>::set(const Array<T>& arr)
   {
      unsigned int rows = arr.rowDim();
      unsigned int cols = arr.colDim();

      if (rows != row_dim || cols != col_dim)
         resize(rows, cols);

      copy(arr);
   }

   template<class T> inline
   void Array<T>::set(const std::vector<std::vector<T> >& arr)
   {
      unsigned int src_row_dim = arr.size();
      unsigned int src_col_dim = 0;

      if (src_row_dim > 0)
         src_col_dim = arr[0].size();

      if (src_row_dim != row_dim || src_col_dim != col_dim)
         resize(src_row_dim, src_col_dim);

      for (unsigned int row_ndx = 0; row_ndx < row_dim; row_ndx++)
         for (unsigned int col_ndx = 0; col_ndx < col_dim; col_ndx++)
            this->at(row_ndx, col_ndx) = arr[row_ndx][col_ndx];
   }

   template<class T> inline
   void Array<T>::resize(unsigned int rows, unsigned int cols, T fill)
   {
      if ((rows == 0 && cols > 0) || (rows > 0 && cols == 0))
      {
         std::ostringstream err_str;
         err_str
            << "Illegal Argument Exception : Array<T>::resize(int,int) - invalid array dimension ("
            << rows << "," << cols << ")";
         throw std::invalid_argument(err_str.str());
      }

      /*
       * Resize std::vector of rows. If this increases the number of rows it should
       * fill in the new rows with the default type of std::vector<T>
       */
      data.resize(rows);

      /*
       * resize the columns of each row with the specified fill vectorize
       */
      for (unsigned int i = 0; i < rows; i++)
         data[i].resize(cols, fill);

      row_dim = rows;
      col_dim = cols;
   }

   template<class T> inline Array<T>& Array<T>::operator=(const T& val)
   {
      set(val);
      return *this;
   }

   template<class T> inline Array<T>& Array<T>::operator=(const Array<T>& arr)
   {
      unsigned int src_row_dim = arr.rowDim();
      unsigned int src_col_dim = arr.colDim();

      if (src_row_dim != row_dim || src_col_dim != col_dim)
      {
         std::ostringstream err_str;
         err_str
            << "Illegal Operand Exception : Array<T>::operator=(Array<T>) - mismatched array dimension :\n"
            << "   lhs:(" << row_dim << "," << col_dim << ")\n" << "   rhs:("
            << src_row_dim << "," << src_col_dim << ")";
         throw std::invalid_argument(err_str.str());
      }

      copy(arr);
      return *this;
   }

/**
 * Array<T>::copy(const Array<T>& arr)
 *
 * Precondition: The dimensionality of the source array matches that
 *    of this array (i.e. validation has been done in the public member
 *    functions).
 */
   template<class T> inline
   void Array<T>::copy(const Array<T>& arr)
   {
      for (unsigned int row_ndx = 0; row_ndx < row_dim; row_ndx++)
         for (unsigned int col_ndx = 0; col_ndx < col_dim; col_ndx++)
            this->at(row_ndx, col_ndx) = arr.at(row_ndx, col_ndx);
   }

   template<class T> inline Array<T>& Array<T>::operator+=(const Array<T>& arr)
   {
      unsigned int src_row_dim = arr.rowDim();
      unsigned int src_col_dim = arr.colDim();

      if (src_row_dim != row_dim || src_col_dim != col_dim)
      {
         std::ostringstream err_str;
         err_str
            << "Illegal Operand Exception : Array<T>::operator+=(Array<T>) - mismatched array dimension :\n"
            << "   lhs:(" << row_dim << "," << col_dim << ")\n" << "   rhs:("
            << src_row_dim << "," << src_col_dim << ")";
         throw std::invalid_argument(err_str.str());
      }

      for (unsigned int row_ndx = 0; row_ndx < src_row_dim; row_ndx++)
         for (unsigned int col_ndx = 0; col_ndx < src_col_dim; col_ndx++)
            this->at(row_ndx, col_ndx) += arr.at(row_ndx, col_ndx);

      return *this;
   }

   template<class T> inline Array<T> operator+(const Array<T>& arr1,
                                               const Array<T>& arr2)
   {
      unsigned int a1_row_dim = arr1.rowDim();
      unsigned int a1_col_dim = arr1.colDim();

      unsigned int a2_row_dim = arr2.rowDim();
      unsigned int a2_col_dim = arr2.colDim();

      if ((a1_row_dim != a2_row_dim || a1_col_dim != a2_col_dim))
      {
         std::ostringstream err_str;
         err_str
            << "Illegal Operand Exception : Array<T>::operator+(Array<T>, Array<T>) - mismatched array dimension :\n"
            << "arg1 :(" << a1_row_dim << "," << a1_col_dim << ")\n"
            << "arg2 :(" << a2_row_dim << "," << a2_col_dim << ")";
         throw std::invalid_argument(err_str.str());
      }

      Array<T> ret_arr(a1_row_dim, a1_col_dim);
      for (unsigned int row_ndx = 0; row_ndx < a1_row_dim; row_ndx++)
         for (unsigned int col_ndx = 0; col_ndx < a1_col_dim; col_ndx++)
            ret_arr.at(row_ndx, col_ndx) = arr1.at(row_ndx, col_ndx)
                                           + arr2.at(row_ndx, col_ndx);

      return ret_arr;
   }

   template<class T>
   class Array3D
   {
   public:

      /* *****************************************
       *   Constructors and configuration
       * *****************************************
       */
      Array3D();
      Array3D(unsigned int xdim, unsigned int ydim, unsigned int zdim);
      void resize(unsigned int xdim, unsigned int ydim, unsigned int zdim, T val =
      T());

      /* *****************************************
       *   Getters/setters
       * *****************************************
       */
      unsigned int XDim() const;
      unsigned int YDim() const;
      unsigned int ZDim() const;

      std::vector<std::vector<T> >& operator[](unsigned int XIdx);
      const std::vector<std::vector<T> >& operator[](unsigned int XIdx) const;

      T& at(unsigned int XIdx, unsigned int YIdx, unsigned int ZIdx);
      const T& at(unsigned int XIdx, unsigned int YIdx, unsigned int ZIdx) const;

      /* *****************************************
       *   Utility functions
       * *****************************************
       */
      void set(T val = T());
      void set(const Array3D<T>& arr);

      Array3D<T>& operator=(const T& val);
      Array3D<T>& operator=(const Array3D<T>& arr);
      Array3D<T>& operator+=(const Array3D<T> arr);

   private:

      void copy(const Array3D<T>& arr);

      unsigned int x_dim;
      unsigned int y_dim;
      unsigned int z_dim;
      std::vector<std::vector<std::vector<T> > > data;
   };

   template<class T> inline Array3D<T>::Array3D()
   {
      x_dim = 0;
      y_dim = 0;
      z_dim = 0;
   }

   template<class T> inline Array3D<T>::Array3D(unsigned int xdim,
                                                unsigned int ydim, unsigned int zdim)
   {
      resize(xdim, ydim, zdim);
   }

   template<class T> inline
   unsigned int Array3D<T>::XDim() const
   {
      return x_dim;
   }

   template<class T> inline
   unsigned int Array3D<T>::YDim() const
   {
      return y_dim;
   }

   template<class T> inline
   unsigned int Array3D<T>::ZDim() const
   {
      return z_dim;
   }

   template<class T> inline std::vector<std::vector<T> >& Array3D<T>::operator[](
      unsigned int XIdx)
   {
      return data[XIdx];
   }

   template<class T> inline const std::vector<std::vector<T> >& Array3D<T>::operator[](
      unsigned int XIdx) const
   {
      return data[XIdx];
   }

   template<class T> inline T& Array3D<T>::at(unsigned int XIdx, unsigned int YIdx,
                                              unsigned int ZIdx)
   {
      return data.at(XIdx).at(YIdx).at(ZIdx);
   }

   template<class T> inline const T& Array3D<T>::at(unsigned int XIdx,
                                                    unsigned int YIdx, unsigned int ZIdx) const
   {
      return data.at(XIdx).at(YIdx).at(ZIdx);
   }

   template<class T> inline
   void Array3D<T>::set(T val)
   {
      for (unsigned int i = 0; i < x_dim; i++)
         for (unsigned int j = 0; j < y_dim; j++)
            data[i][j].assign(z_dim, val);
   }

   template<class T> inline
   void Array3D<T>::set(const Array3D<T>& arr)
   {
      unsigned int src_x_dim = arr.XDim();
      unsigned int src_y_dim = arr.YDim();
      unsigned int src_z_dim = arr.ZDim();

      if ((x_dim != src_x_dim || y_dim != src_y_dim || z_dim != src_z_dim))
         resize(src_x_dim, src_y_dim, src_z_dim);

      copy(arr);
   }

   template<class T> inline
   void Array3D<T>::resize(unsigned int xdim, unsigned int ydim, unsigned int zdim,
                           T fill)
   {
      if (xdim == 0 || ydim == 0 || zdim == 0)
      {
         std::ostringstream err_str;
         err_str << "Array3D<T>::resize(int,int,int) - invalid array dimension ("
                 << x_dim << "," << y_dim << "," << z_dim << ")";
         throw std::invalid_argument(err_str.str());
      }

      x_dim = xdim;
      y_dim = ydim;
      z_dim = zdim;

      /*
       * Resize std::vector of rows. If this increases the number of rows it should
       * fill in the new rows with the default type of std::vector<T>
       */
      data.resize(xdim);

      /*
       * resize the columns of each row with the specified fill vectorize
       */
      for (unsigned int i = 0; i < x_dim; i++)
      {
         data[i].resize(y_dim);
         for (unsigned int j = 0; j < y_dim; j++)
            data[i][j].resize(z_dim, fill);
      }
   }

   template<class T> inline Array3D<T>& Array3D<T>::operator=(const T& val)
   {
      set(val);
      return *this;
   }

   template<class T> inline Array3D<T>& Array3D<T>::operator=(
      const Array3D<T>& arr)
   {
      unsigned int src_x_dim = arr.XDim();
      unsigned int src_y_dim = arr.YDim();
      unsigned int src_z_dim = arr.ZDim();

      if ((x_dim != src_x_dim || y_dim != src_y_dim || z_dim != src_z_dim))
      {
         std::ostringstream err_str;
         err_str
            << "Illegal Operand Exception : Array3D<T>::operator=(Array3D<T>) - mismatched array dimension :\n"
            << "   lhs:(" << x_dim << "," << y_dim << "," << z_dim << ")\n"
            << "   rhs:(" << src_x_dim << "," << src_y_dim << "," << src_z_dim
            << ")";
         throw std::invalid_argument(err_str.str());
      }

      copy(arr);
      return *this;
   }

/**
 * Array3D<T>::copy(const Array3D<T>& arr)
 *
 * Precondition: The dimensionality of the source array matches that
 *    of this array (i.e. validation has been done in the public member
 *    functions).
 */
   template<class T> inline
   void Array3D<T>::copy(const Array3D<T>& arr)
   {
      for (unsigned int x_ndx = 0; x_ndx < x_dim; x_ndx++)
         for (unsigned int y_ndx = 0; y_ndx < y_dim; y_ndx++)
            data[x_ndx][y_ndx] = arr[x_ndx][y_ndx];
   }

   template<class T> inline Array3D<T>& Array3D<T>::operator+=(
      const Array3D<T> arr)
   {
      unsigned int src_x_dim = arr.XDim();
      unsigned int src_y_dim = arr.YDim();
      unsigned int src_z_dim = arr.ZDim();

      if ((x_dim != src_x_dim || y_dim != src_y_dim || z_dim != src_z_dim))
      {
         std::ostringstream err_str;
         err_str
            << "Illegal Operand Exception : Array3D<T>::operator+=(Array3D<T>) - mismatched array dimension :\n"
            << "   lhs:(" << x_dim << "," << y_dim << "," << z_dim << ")\n"
            << "   rhs:(" << src_x_dim << "," << src_y_dim << "," << src_z_dim
            << ")";
         throw std::invalid_argument(err_str.str());
      }

      for (unsigned int x_ndx = 0; x_ndx < x_dim; x_ndx++)
         for (unsigned int y_ndx = 0; y_ndx < y_dim; y_ndx++)
            for (unsigned int z_ndx = 0; z_ndx < z_dim; z_ndx++)
               this->at(x_ndx, y_ndx, z_ndx) += arr.at(x_ndx, y_ndx, z_ndx);

      return *this;
   }

   template<class T> inline Array3D<T> operator+(const Array3D<T>& arr1,
                                                 const Array3D<T>& arr2)
   {
      unsigned int a1_x_dim = arr1.XDim();
      unsigned int a1_y_dim = arr1.YDim();
      unsigned int a1_z_dim = arr1.ZDim();

      unsigned int a2_x_dim = arr2.XDim();
      unsigned int a2_y_dim = arr2.YDim();
      unsigned int a2_z_dim = arr2.ZDim();

      if ((a1_x_dim != a2_x_dim || a1_y_dim != a2_y_dim || a1_z_dim != a2_z_dim))
      {
         std::ostringstream err_str;
         err_str
            << "Illegal Operand Exception : Array3D<T>::operator+(Array3D<T>, Array3D<T>) - mismatched array dimension :\n"
            << "arg1 :(" << a1_x_dim << "," << a1_y_dim << "," << a1_z_dim
            << ")\n" << "arg2 :(" << a2_x_dim << "," << a2_y_dim << ","
            << a2_z_dim << ")";
         throw std::invalid_argument(err_str.str());
      }

      Array3D<T> ret_arr(a1_x_dim, a1_y_dim, a1_z_dim);
      for (unsigned int x_ndx = 0; x_ndx < a1_x_dim; x_ndx++)
         for (unsigned int y_ndx = 0; y_ndx < a1_y_dim; y_ndx++)
            for (unsigned int z_ndx = 0; z_ndx < a1_z_dim; z_ndx++)
               ret_arr.at(x_ndx, y_ndx, z_ndx) = arr1.at(x_ndx, y_ndx, z_ndx)
                                                 + arr2.at(x_ndx, y_ndx, z_ndx);

      return ret_arr;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_ARRAY_H_ */
