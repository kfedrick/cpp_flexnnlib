//
// Created by kfedrick on 5/31/19.
//

#include "Datum.h"

#include <sstream>
#include <map>
#include <functional>
#include <algorithm>

using flexnnet::Datum;

Datum::Datum(void) {}

Datum::Datum(const std::map< std::string, std::valarray<double> >& _values)
{
   set (_values);
}


Datum::~Datum()
{
}

const std::set<std::string>& Datum::key_set() const
{
   //hash<std::set_weights<std::string>>(keyset);
   return keyset;
}

size_t Datum::hash()
{
   std::hash<unsigned long> size_t_hash;
   std::hash<std::string> string_hash;

   size_t hashval = 0;
   for (auto& item : fields)
      hashval = hashval ^ string_hash(item.first) ^ size_t_hash(item.second.index) ^ size_t_hash(item.second.len);

   return hashval;
}

Datum& Datum::operator=(const Datum& _datum)
{
   return copy(_datum);
}

Datum& Datum::copy(const Datum& _datum)
{
   fields = _datum.fields;
   data = _datum.data;
   keyset = _datum.keyset;
   virtual_array = _datum.virtual_array;
   stale = _datum.stale;
   fields_hashval = _datum.fields_hashval;
}



void Datum::set (const std::map<std::string, std::valarray<double> > &_values)
{
   // Clear existing Datum keyset and reset it using the specified map
   keyset.clear();
   for (auto& item : _values)
      keyset.insert(item.first);

   // Clear and set_weights fields and data items
   fields.clear();
   data.resize(_values.size());
   data.shrink_to_fit();

   size_t ndx = 0;
   for (auto& key : keyset)
   {
      fields[key] = { .index=ndx, .len=_values.at(key).size() };
      data[ndx++] = _values.at(key);
   }

   resize_virtual();

   fields_hashval = hash();
   stale = true;
}

void Datum::set (const std::string _key, const std::valarray<double> &_value)
{
   // If key is doesn't exist, throw exception
   if (fields.count(_key) == 0)
   {
      std::ostringstream err_str;
      err_str
         << "Error : Datum.set_weights() - key value " << _key << " doesn't exists.\n";
      throw std::invalid_argument (err_str.str ());
   }

   // If new value array is not the correct size, throw exception
   if (fields[_key].len != _value.size())
   {
      std::ostringstream err_str;
      err_str
         << "Error : Datum.set_weights() - new valarray size " << _value.size()
         << "doesn't match expected size " << fields[_key].len << ".\n";
      throw std::invalid_argument (err_str.str ());
   }

   // Set new value
   data[fields[_key].index] = _value;

}

void Datum::resize_virtual() const
{
   size_t sz = 0;
   for (auto& item : data)
      sz += item.size();

   virtual_array.resize(sz);
}


void Datum::coelesce() const
{
   if (!stale)
      return;

   size_t vndx = 0;
   for (auto& item : data)
      for (auto i=0; i<item.size(); i++)
         virtual_array[vndx++] = item[i];

      stale = false;
}

void Datum::insert (const std::string &_index, const std::valarray<double> &_value)
{
   // If key is already there, throw exception
   if (fields.count(_index) == 1)
   {
      std::ostringstream err_str;
      err_str
         << "Error : Datum.insert() - key value " << _index << " already exists.\n";
      throw std::invalid_argument (err_str.str ());
   }

   // Insert new value key
   keyset.insert(_index);

   // Adjust all of the field and data entries
   std::vector<std::valarray<double>> newdata(data.size() + 1);

   size_t ndx = 0;
   for (auto& key : keyset)
   {
      // if field entry already exist, copy existing data to newdata and adjust the ndx
      if (fields.count(key) == 1)
      {
         newdata[ndx] = data[fields[key].index];
         fields[key].index = ndx;
      }
      else
      {
         fields[key] = { .index=ndx, .len=_value.size() };
         newdata[ndx] = _value;
      }

      ndx++;
   }

   // Reassign data to updated data
   data = newdata;

   resize_virtual();

   fields_hashval = hash();
   stale = true;
}
