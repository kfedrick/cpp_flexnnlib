//
// Created by kfedrick on 9/9/19.
//

#ifndef FLEX_NEURALNET_ENUMERATEDDATASET_H_
#define FLEX_NEURALNET_ENUMERATEDDATASET_H_

#include <set>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cxxabi.h>
#include <Globals.h>
#include <JSONStream.h>
#include <document.h>
#include <istream>
#include <istreamwrapper.h>
#include <stringbuffer.h>

#include <flexnnet.h>


// Forward declaration for CartesianCoord
namespace flexnnet {    template<class _InTyp, class _OutTyp>
    class EnumeratedDataSet; }

// Forward declarations for stream operators
template<class _InTyp, class _OutTyp>
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::Exemplar<_InTyp, _OutTyp>& _exemplar);

template<class _InTyp, class _OutTyp>
std::istream& operator>>(std::istream& _istrm, flexnnet::Exemplar<_InTyp, _OutTyp>& _exemplar);

template<class _InTyp, class _OutTyp>
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::EnumeratedDataSet<_InTyp, _OutTyp>& _dataset);

template<class _InTyp, class _OutTyp>
std::istream& operator>>(std::istream& _istrm, flexnnet::EnumeratedDataSet<_InTyp, _OutTyp>& _dataset);

namespace flexnnet
{
   template<class _InTyp, class _OutTyp>
   class EnumeratedDataSet
   {
      using _DataTyp = Exemplar<_InTyp,_OutTyp>;
      using order_iterator = std::vector<int>::iterator;
      using order_const_iterator = std::vector<int>::const_iterator;

   public:
      void
      reset(void);

      /**
       * Return the current number of exemplars in the data set.
       * @return
       */
      size_t
      size(void) const;

      /**
       * Clear all exemplars from the data set.
       */
      void
      clear(void);

      /**
       * Add a new exemplar to the data set.
       * @param _data
       */
      void
      push_back(const _DataTyp& _data);

      /**
       * Create a new permutation for the order in which the iterator
       * traverses the exemplars in the data set.
       */
      void
      randomize_order(void) const;

      /**
       * Reset the normal presentation order in which the iterator
       * traverses the exemplars in the data set.
       */
      void
      normalize_order(void) const;

      friend std::ostream& ::operator<<<_InTyp,_OutTyp>(std::ostream& _ostrm, const flexnnet::Exemplar<_InTyp, _OutTyp>& _exemplar);

      friend std::istream& ::operator>><_InTyp,_OutTyp>(std::istream& _istrm, flexnnet::Exemplar<_InTyp, _OutTyp>& _exemplar);

      friend std::ostream& ::operator<<<_InTyp,_OutTyp>(std::ostream& _ostrm, const EnumeratedDataSet<_InTyp, _OutTyp>& _dataset);

      template<typename _InTyp1, typename _OutTyp1>
      friend std::istream& ::operator>>(std::istream& _istrm, EnumeratedDataSet<_InTyp1, _OutTyp1>& _dataset);

      /*
       * Public iterators
       */
   public:

      class iterator : public std::iterator<std::forward_iterator_tag,
                                            _DataTyp, int, _DataTyp*, _DataTyp*>
      {
      public:
         typedef iterator self_type;
         typedef _DataTyp value_type;
         typedef _DataTyp& reference;
         typedef _DataTyp* pointer;
         typedef std::forward_iterator_tag iterator_category;
         typedef int difference_type;
         iterator(std::vector<_DataTyp>& _data, order_iterator ptr) : data(_data), it(ptr)
         {
         }

         self_type
         operator++()
         {
            it++;
            return *this;
         }
         self_type
         operator++(int junk)
         {
            it++;
            return *this;
         }
         reference
         operator*()
         { return data[*it]; }

         reference
         operator->()
         { return data[*it]; }

         bool
         operator==(const self_type& rhs)
         { return it == rhs.it; }

         bool
         operator!=(const self_type& rhs)
         { return it != rhs.it; }
      private:
         order_iterator it;
         std::vector<_DataTyp>& data;
      };

      class const_iterator : public std::iterator<std::forward_iterator_tag,
                                                  _DataTyp,
                                                  int,
                                                  const _DataTyp*,
                                                  const _DataTyp*>
      {
      public:
         typedef const_iterator self_type;
         typedef _DataTyp value_type;
         typedef const _DataTyp& reference;
         typedef const _DataTyp* pointer;
         typedef int difference_type;
         typedef std::forward_iterator_tag iterator_category;
         const_iterator(const std::vector<_DataTyp>& _data, order_const_iterator ptr) : data(_data), it(ptr)
         {
         }

         self_type
         operator++()
         {
            it++;
            return *this;
         }
         self_type
         operator++(int junk)
         {
            it++;
            return *this;
         }
         const reference
         operator*()
         { return data[*it]; }

         const reference
         operator->()
         { return data[*it]; }

         bool
         operator==(const self_type& rhs)
         { return it == rhs.it; }

         bool
         operator!=(const self_type& rhs)
         { return it != rhs.it; }
      private:
         order_const_iterator it;
         const std::vector<_DataTyp>& data;
      };

      iterator
      begin()
      {
         return iterator(data, presentation_order.begin());
      }

      iterator
      end()
      {
         return iterator(data, presentation_order.end());
      }

      const_iterator
      begin() const
      {
         return const_iterator(data, presentation_order.begin());
      }

      const_iterator
      end() const
      {
         return const_iterator(data, presentation_order.end());
      }

   private:
      int urand(int n);
      void write_exemplar(std::ostream& _ostrm, const _DataTyp& _exemplar) const;

   private:
      std::vector<_DataTyp> data;
      mutable std::vector<int> presentation_order;

   };



   template<class _InTyp, class _OutTyp>
   inline size_t
   EnumeratedDataSet<_InTyp, _OutTyp>::size(void) const
   {
      return data.size();
   }

   template<class _InTyp, class _OutTyp>
   inline void
   EnumeratedDataSet<_InTyp, _OutTyp>::clear(void)
   {
      data.clear();
      presentation_order.clear();
   }

   template<class _InTyp, class _OutTyp>
   inline void
   EnumeratedDataSet<_InTyp, _OutTyp>::push_back(const _DataTyp& _data)
   {
      data.push_back(_data);
      presentation_order.push_back(data.size()-1);
   }

   template<class _InTyp, class _OutTyp>
   inline void
   EnumeratedDataSet<_InTyp, _OutTyp>::randomize_order(void) const
   {
      unsigned int new_ndx, temp;
      unsigned int sz = presentation_order.size();

      for (unsigned int rounds = 0; rounds < 2; rounds++)
      {
         for (unsigned int ndx = 0; ndx < sz; ndx++)
         {
            new_ndx = urand(sz);

            temp = presentation_order[new_ndx];
            presentation_order[new_ndx] = presentation_order[ndx];
            presentation_order[ndx] = temp;
         }
      }
   };

   template<class _InTyp, class _OutTyp>
   inline void
   EnumeratedDataSet<_InTyp, _OutTyp>::normalize_order(void) const
   {
      presentation_order.resize(data.size());
      for (size_t i=0; i<presentation_order.size(); i++)
         presentation_order[i] = i;
   };

   template<class _InTyp, class _OutTyp>
   inline int
   EnumeratedDataSet<_InTyp, _OutTyp>::urand(int n)
   {
      if (n == 0)
         return 0;

      int top = ((((RAND_MAX - n) + 1) / n) * n - 1) + n;
      int r;
      do
      {
         r = rand();
      }
      while (r > top);
      return (r % n);
   }
}

#endif //FLEX_NEURALNET_ENUMERATEDDATASET_H_
