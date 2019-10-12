//
// Created by kfedrick on 9/9/19.
//

#ifndef FLEX_NEURALNET_DATASET_H_
#define FLEX_NEURALNET_DATASET_H_

#include <set>
#include "Exemplar.h"

namespace flexnnet
{
   template<class _InType, class _OutType, template<class, class> class _Sample>
   class DataSet : public std::set<_Sample<_InType, _OutType> >
   {
      void initialize(void)
      {};
      void randomize_order(void)
      {};

      /*
   public:

      typedef int size_type;

      class iterator
      {
      public:
         typedef iterator self_type;
         typedef _DataType value_type;
         typedef _DataType& reference;
         typedef _DataType* pointer;
         typedef std::forward_iterator_tag iterator_category;
         typedef int difference_type;
         iterator(pointer ptr) : ptr_(ptr) { }
         self_type operator++() { self_type i = *this; ptr_++; return i; }
         self_type operator++(int junk) { ptr_++; return *this; }
         reference operator*() { return *ptr_; }
         pointer operator->() { return ptr_; }
         bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
         bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
      private:
         pointer ptr_;
      };

      class const_iterator
      {
      public:
         typedef const_iterator self_type;
         typedef _DataType value_type;
         typedef _DataType& reference;
         typedef _DataType* pointer;
         typedef int difference_type;
         typedef std::forward_iterator_tag iterator_category;
         const_iterator(pointer ptr) : ptr_(ptr) { }
         self_type operator++() { self_type i = *this; ptr_++; return i; }
         self_type operator++(int junk) { ptr_++; return *this; }
         const reference operator*() { return *ptr_; }
         const pointer operator->() { return ptr_; }
         bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
         bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
      private:
         pointer ptr_;
      };

      iterator begin()
      {
         return iterator(data_);
      }

      iterator end()
      {
         return iterator(data_ + size_);
      }

      const_iterator begin() const
      {
         return const_iterator(data_);
      }

      const_iterator end() const
      {
         return const_iterator(data_ + size_);
      }

   private:
      _DataType* data;
       */

   };

}

#endif //FLEX_NEURALNET_DATASET_H_
