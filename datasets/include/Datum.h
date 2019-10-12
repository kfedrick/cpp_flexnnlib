//
// Created by kfedrick on 5/31/19.
//

#ifndef FLEX_NEURALNET_DATUM_H_
#define FLEX_NEURALNET_DATUM_H_

#include <map>
#include <set>
#include <string>
#include <valarray>

namespace flexnnet
{
   class Datum
   {
   private:
      struct Entry
      {
         size_t index;
         size_t len;
      };

   public:
      Datum();
      Datum(const std::map<std::string, std::valarray<double> >& _values);

      virtual ~Datum();

      size_t size() const;
      size_t count() const;
      size_t hashval() const;

      const std::set<std::string>& key_set() const;
      size_t index(const std::string& _index) const;

      Datum& operator=(const Datum& _datum);
      Datum& operator=(const std::map<std::string, std::valarray<double> >& _values);

      void insert(const std::string& _index, const std::valarray<double>& _value);
      void set(const std::map<std::string, std::valarray<double> >& _values);
      void set(const std::string _key, const std::valarray<double>& _value);

      const std::valarray<double>& vectorize(void) const
      {
         return (*this)();
      }

      const std::valarray<double>& operator()(void) const;
      std::valarray<double>& operator()(void);

      const std::valarray<double>& operator[](const std::string& _index) const;
      const std::valarray<double>& operator[](size_t _index) const;
      const std::valarray<double>& at(size_t _index) const;

   public:
      const std::set<std::string>& const_keyset_ref = keyset;
      const std::map<const std::string, Entry>& const_fields_ref = fields;

   private:
      Datum& copy(const Datum& _datum);

      void resize_virtual() const;
      void coelesce() const;

      size_t hash();

   private:
      mutable size_t fields_hashval;
      mutable bool stale;
      mutable std::valarray<double> virtual_array;
      mutable std::set<std::string> keyset;

      std::map<const std::string, Entry> fields;
      std::vector<std::valarray<double> > data;
   };

   inline size_t Datum::size() const
   {
      return virtual_array.size();
   }

   inline size_t Datum::count() const
   {
      return data.size();
   }

   inline size_t Datum::hashval() const
   {
      return fields_hashval;
   }

   inline size_t Datum::index(const std::string& _index) const
   {
      return fields.at(_index).index;
   }

   inline std::valarray<double>& Datum::operator()(void)
   {
      if (stale)
         coelesce();

      return virtual_array;
   }

   inline const std::valarray<double>& Datum::operator()(void) const
   {
      if (stale)
         coelesce();

      return virtual_array;
   }

   inline const std::valarray<double>& Datum::operator[](const std::string& _index) const
   {
      return data[fields.at(_index).index];
   }

   inline const std::valarray<double>& Datum::operator[](size_t _index) const
   {
      return data[_index];
   }

   inline const std::valarray<double>& Datum::at(size_t _index) const
   {
      return data.at(_index);
   }

}

#endif //_DATUM_H_
