/*
 * Array#D.h
 *
 *  Created on: Jan 30, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ARRAY3D_H_
#define FLEX_NEURALNET_ARRAY3D_H_

#include <vector>
#include <valarray>
#include <sstream>
#include <stdexcept>
#include "Reinforcement.h"

namespace flexnnet
{
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
       * resize the columns of each row with the specified fill value
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

#endif /* FLEX_NEURALNET_ARRAY2_H_ */

