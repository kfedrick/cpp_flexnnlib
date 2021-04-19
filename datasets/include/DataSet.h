//
// Created by kfedrick on 9/9/19.
//

#ifndef FLEX_NEURALNET_DATASET_H_
#define FLEX_NEURALNET_DATASET_H_

#include <flexnnet.h>
#include <vector>
#include <iostream>
#include <random>

#include <Exemplar.h>

// Forward declaration for CartesianCoord
namespace flexnnet
{
   template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
   class DataSet;
}

// Forward declarations for stream operators
template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
std::ostream&
operator<<(std::ostream& _ostrm, const ItemTyp<InTyp, OutTyp>& _item);

template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
std::istream&
operator>>(std::istream& _istrm, ItemTyp<InTyp, OutTyp>& _item);

template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
std::ostream&
operator<<(std::ostream& _ostrm, const flexnnet::DataSet<InTyp, OutTyp, ItemTyp>& _dataset);

template<class InTyp, class TgtTyp, template<class,class> class ItemTyp>
std::istream&
operator>>(std::istream& _istrm, flexnnet::DataSet<InTyp, TgtTyp, ItemTyp>& _dataset);

namespace flexnnet
{
   template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
   class DataSet
   {
      using ExemplarTyp = Exemplar<InTyp, OutTyp>;
      using order_iterator = std::vector<int>::iterator;
      using order_const_iterator = std::vector<int>::const_iterator;

   public:
      DataSet(void);

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
      push_back(const ItemTyp<InTyp,OutTyp>& _data);

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

      friend std::ostream&::operator
      <<<InTyp,OutTyp,ItemTyp>(
      std::ostream& _ostrm,
      const ItemTyp<InTyp, OutTyp>& _exemplar
      );

      friend std::istream&
      ::operator>><InTyp, OutTyp>(std::istream& _istrm, flexnnet::Exemplar<InTyp,
                                                                           OutTyp>& _exemplar);

      friend std::ostream&
         ::operator
      <<<InTyp,OutTyp,ItemTyp>(
      std::ostream& _ostrm,
      const DataSet<InTyp, OutTyp, ItemTyp>& _dataset
      );

      friend std::istream&
      ::operator>><InTyp,OutTyp,ItemTyp>(std::istream& _istrm, DataSet<InTyp, OutTyp, ItemTyp>& _dataset);

      /*
       * Public iterators
       */
   public:

      class iterator : public std::iterator<std::forward_iterator_tag,
                                            ItemTyp<InTyp,OutTyp>, int, ItemTyp<InTyp,OutTyp>*, ItemTyp<InTyp,OutTyp>*>
      {
      public:
         typedef iterator self_type;
         typedef ItemTyp<InTyp,OutTyp> value_type;
         typedef ItemTyp<InTyp,OutTyp>& reference;
         typedef ItemTyp<InTyp,OutTyp>* pointer;
         typedef std::forward_iterator_tag iterator_category;
         typedef int difference_type;
         iterator(std::vector<ItemTyp<InTyp,OutTyp>>& _data, order_iterator ptr)
            : data(_data), it(ptr)
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
         std::vector<ItemTyp<InTyp,OutTyp>>& data;
      };

      class const_iterator : public std::iterator<std::forward_iterator_tag,
                                                  ItemTyp<InTyp,OutTyp>,
                                                  int,
                                                  const ItemTyp<InTyp,OutTyp>*,
                                                  const ItemTyp<InTyp,OutTyp>*>
      {
      public:
         typedef const_iterator self_type;
         typedef ItemTyp<InTyp,OutTyp> value_type;
         typedef const ItemTyp<InTyp,OutTyp>& reference;
         typedef const ItemTyp<InTyp,OutTyp>* pointer;
         typedef int difference_type;
         typedef std::forward_iterator_tag iterator_category;
         const_iterator(const std::vector<ItemTyp<InTyp,OutTyp>>& _data, order_const_iterator ptr)
            : data(_data), it(ptr)
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
         const std::vector<ItemTyp<InTyp,OutTyp>>& data;
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
      //void
      //write_exemplar(std::ostream& _ostrm, const ExemplarTyp& _exemplar) const;

   private:
      std::vector<ItemTyp<InTyp, OutTyp>> data;
      mutable std::vector<int> presentation_order;

      //mutable std::default_random_engine rand_engine;
      mutable std::mt19937_64 rand_engine;

   };

   template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
   inline
   DataSet<InTyp, OutTyp, ItemTyp>::DataSet(void)
   {
      std::random_device r;
      std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
      rand_engine.seed(seed2);
   }

   template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
   inline size_t
   DataSet<InTyp, OutTyp, ItemTyp>::size(void) const
   {
      return data.size();
   }

   template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
   inline void
   DataSet<InTyp, OutTyp, ItemTyp>::clear(void)
   {
      data.clear();
      presentation_order.clear();
   }

   template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
   inline void
   DataSet<InTyp, OutTyp, ItemTyp>::push_back(const ItemTyp<InTyp,OutTyp>& _data)
   {
      data.push_back(_data);
      presentation_order.push_back(data.size() - 1);
   }

   template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
   inline void
   DataSet<InTyp, OutTyp, ItemTyp>::randomize_order(void) const
   {
      unsigned int new_ndx, temp;
      unsigned int sz = presentation_order.size();

      std::uniform_int_distribution<int> uniform_dist(1, sz - 1);
      for (unsigned int ndx = 0; ndx < sz; ndx++)
      {
         new_ndx = uniform_dist(rand_engine);

         // Don't swap if new index is less than or equal to
         // current index to avoid double swapping.
         if (new_ndx <= ndx)
            continue;

         // Swap
         temp = presentation_order[new_ndx];
         presentation_order[new_ndx] = presentation_order[ndx];
         presentation_order[ndx] = temp;
      }
   };

   template<class InTyp, class OutTyp, template<class,class> class ItemTyp>
   inline void
   DataSet<InTyp, OutTyp, ItemTyp>::normalize_order(void) const
   {
      presentation_order.resize(data.size());
      for (size_t i = 0; i < presentation_order.size(); i++)
         presentation_order[i] = i;
   };
}

#endif //FLEX_NEURALNET_DATASET_H_
