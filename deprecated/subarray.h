//
// Created by kfedrick on 6/1/19.
//

#ifndef _SUBARRAY_H_
#define _SUBARRAY_H_

#include <array>
#include <stddef.h>

namespace flexnnet
{

   template<class T> class basic_bidirectional_iterator
   {
   public:
      typedef basic_bidirectional_iterator<T>& self_ref;

      using iterator_category = std::forward_iterator_tag;
      using value_type = T;
      using difference_type = T;
      using pointer = T *;
      using reference = T &;

      basic_bidirectional_iterator(pointer _ptr) : ptr(_ptr) { }
      self_ref operator++() { self_ref i = *this; ptr++; return i; }
      self_ref operator++(int junk) { ptr++; return *this; }
      self_ref operator--() { self_ref i = *this; ptr--; return i; }
      self_ref operator--(int junk) { ptr--; return *this; }
      reference operator*() { return *ptr; }
      pointer operator->() { return ptr; }
      bool operator==(const self_ref& rhs) { return ptr == rhs.ptr; }
      bool operator!=(const self_ref& rhs) { return ptr != rhs.ptr; }
      ptrdiff_t operator-(const basic_bidirectional_iterator<T>& _iter) { return std::distance(ptr,_iter.ptr);}
   private:
      pointer ptr;
   };

   template<class T> class const_basic_bidirectional_iterator
   {
   public:
      typedef basic_bidirectional_iterator<T>& self_ref;

      using iterator_category = std::forward_iterator_tag;
      using value_type = T;
      using difference_type = T;
      using pointer = T *;
      using const_pointer = T *;
      using const_reference = T &;

      const_basic_bidirectional_iterator(const_pointer _ptr, size_t _sz) : ptr(_ptr) { }
      self_ref operator++() { self_ref i = *this; ptr++; return i; }
      self_ref operator++(int junk) { ptr++; return *this; }
      self_ref operator--() { self_ref i = *this; ptr--; return i; }
      self_ref operator--(int junk) { ptr--; return *this; }
      const_reference operator*() { return *ptr; }
      const_pointer operator->() { return ptr; }
      bool operator==(const self_ref& rhs) { return ptr == rhs.ptr; }
      bool operator!=(const self_ref& rhs) { return ptr != rhs.ptr; }
      self_ref begin() { return self_ref(ptr, 0); }
      self_ref end() { return self_ref(ptr, sz); }
      ptrdiff_t operator-(const basic_bidirectional_iterator<T>& _iter) { return std::distance(ptr,_iter.ptr);}
   private:
      pointer ptr;
      size_t sz;
   };

   template<class T>
   class subarray
   {
   public:
      subarray (T *_array_ref, size_t _offset, size_t _sz);

      // types:
      typedef T &reference;
      typedef const T &const_reference;
      //typedef subarrayIterator<T> iterator;
      typedef basic_bidirectional_iterator<T> iterator;
      typedef const_basic_bidirectional_iterator<T> const_iterator;
      //typedef std::<T> const_iterator;
      typedef size_t size_type;
      typedef ptrdiff_t difference_type;
      typedef T value_type;
      typedef T *pointer;
      typedef const T *const_pointer;
      typedef std::reverse_iterator<iterator> reverse_iterator;
      //typedef std::const_reverse_iterator<const_iterator> const_reverse_iterator;

      // no explicit construct/copy/destroy for aggregate type

      constexpr void fill (const T &u);
      constexpr void swap (subarray<T> &) noexcept;

      // iterators:
      iterator begin () noexcept;
      const_iterator begin () const noexcept;
      iterator end () noexcept;
      const_iterator end () const noexcept;

      //constexpr reverse_iterator rbegin () noexcept;
      //constexpr const_reverse_iterator rbegin () const noexcept;
      //constexpr reverse_iterator rend () noexcept;
      //constexpr const_reverse_iterator rend () const noexcept;

      //constexpr const_iterator cbegin () const noexcept;
      //constexpr const_iterator cend () const noexcept;
      //constexpr const_reverse_iterator crbegin () const noexcept;
      //constexpr const_reverse_iterator crend () const noexcept;

      // capacity:
      constexpr size_type size () const noexcept;
      constexpr size_type max_size () const noexcept;
      constexpr bool empty () const noexcept;

      // element access:
      const reference operator[] (size_type n);
      const const_reference operator[] (size_type n) const;
      const const_reference at (size_type n) const;
      const reference at (size_type n);
      const reference front ();
      const const_reference front () const;
      const reference back ();
      const const_reference back () const;

      const T *data () noexcept;
      constexpr const T *data () const noexcept;

   private:
      size_t offset;
      size_t vsize;
      T *elems;
   };

   template<class T> subarray<T>::subarray (T *_array_ref, size_t _offset, size_t _sz) : elems (_array_ref)
   {
      offset = _offset;
      vsize = _sz;
   }

   template<class T> typename subarray<T>::iterator subarray<T>::begin () noexcept
   {
      return iterator(this->elems+offset);
   }

   template<class T> typename subarray<T>::iterator subarray<T>::end () noexcept
   {
      return iterator(this->elems+offset+vsize);
   }

   template<class T> constexpr typename subarray<T>::size_type subarray<T>::size () const noexcept
   {
      return vsize;
   }

   template<class T> constexpr bool subarray<T>::empty () const noexcept
   {
      return vsize==0;
   }


   template<class T> typename subarray<T>::reference subarray<T>::operator[] (size_type n)
   {
      return elems[n];
   }

   template<class T> typename subarray<T>::const_reference subarray<T>::operator[] (size_type n) const
   {
      return elems[n];
   }

   template<class T> typename subarray<T>::const_reference subarray<T>::at (size_type n) const
   {
      if (n>vsize)
         throw std::out_of_range(0);
      return elems[n];
   }

   template<class T> typename subarray<T>::reference subarray<T>::at (size_type n)
   {
      if (n>vsize)
         throw std::out_of_range(0);
      return elems[n];
   }

   template<class T> const T* subarray<T>::data () noexcept
   {
      return elems;
   }

   template<class T> constexpr const T* subarray<T>::data () const noexcept
   {
      return elems;
   }
}

#endif //_SUBARRAY_H_
