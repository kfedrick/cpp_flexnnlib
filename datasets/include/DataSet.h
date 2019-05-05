/*
 * DataSet.h
 *
 *  Created on: Mar 26, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_DATASET_H_
#define FLEX_NEURALNET_DATASET_H_

#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

namespace flex_neuralnet
{

template <class _DataType>
class DataSet : public vector<_DataType>
{
public:
   DataSet();
   DataSet(int sz);
   DataSet(unsigned int sz);

   /*
    * There is a hidden cost involved with this that I'd rather avoid. Whenever a
    * vector<_DataType> is passed into a function expecting a DataSet<_DataType> it
    * quietly instantiates a temporary DataSet<T> incurring a construction and copy
    * cost.
    *
   DataSet(const vector<_DataType>& vec);
   */

   DataSet<_DataType>& operator=(const vector<_DataType>& vec);

   void toFile(const string& _fname);
   void fromFile(const string& _fname);

   void permute();
   void permute(vector<unsigned int>& _permutation);

private:
   int urand(int n);
};


template <class _DataType>
DataSet<_DataType>::DataSet() : vector<_DataType>() {}

template <class _DataType>
DataSet<_DataType>::DataSet(int sz) : vector<_DataType>(sz) {}

template <class _DataType>
DataSet<_DataType>::DataSet(unsigned int sz) : vector<_DataType>(sz) {}


/*
 * There is a hidden cost involved with this that I'd rather avoid. Whenever a
 * vector<_DataType> is passed into a function expecting a DataSet<_DataType> it
 * quietly instantiates a temporary DataSet<T> incurring a construction and copy
 * cost.
 *
template <class _DataType>
DataSet<_DataType>::DataSet(const vector<_DataType>& vec) : vector<_DataType>(vec) {}
*/

template <class _DataType>
DataSet<_DataType>& DataSet<_DataType>::operator=(const vector<_DataType>& vec)
{
   this->assign(vec.begin(), vec.end());
   return *this;
}

template <class _DataType>
void DataSet<_DataType>::toFile(const string& _fname)
{
   fstream fs;
   fs.open(_fname.c_str(), fstream::out | fstream::binary);

   unsigned int sz = this->size();
   fs.write((char*) &sz, sizeof(unsigned int));

   for (unsigned int ndx=0; ndx < sz; ndx++)
      this->at(ndx).toFile(fs);

   fs.close();
}

template <class _DataType>
void DataSet<_DataType>::fromFile(const string& _fname)
{
   this->clear();

   fstream fs;
   fs.open(_fname.c_str(), fstream::in | fstream::binary);

   unsigned int sz = this->size();
   fs.read((char*) &sz, sizeof(unsigned int));

   _DataType data;
   for (unsigned int ndx=0; ndx < sz; ndx++)
   {
      data.fromFile(fs);
      this->push_back(data);
   }

   fs.close();
}

template <class _DataType>
int DataSet<_DataType>::urand(int n)
{
   if (n == 0)
      return 0;

   int top = ((((RAND_MAX - n) + 1) / n) * n - 1) + n;
   int r;
   do
   {
      r = rand();
   } while (r > top);
   return (r % n);
}

template <class _DataType>
void DataSet<_DataType>::permute()
{
   _DataType temp_data;
   unsigned int new_ndx;
   unsigned int sz = this->size();

   for (unsigned int rounds = 0; rounds < 2; rounds++)
   {
      for (unsigned int ndx = 0; ndx < sz; ndx++)
      {
         new_ndx = urand(sz);

         temp_data = (*this)[new_ndx];
         (*this)[new_ndx] = (*this)[ndx];
         (*this)[ndx] = temp_data;
      }
   }
}

template <class _DataType>
void DataSet<_DataType>::permute(vector<unsigned int>& _permutation)
{
   unsigned int temp_ndx;
   _DataType temp;
   unsigned int new_ndx;
   unsigned int sz = this->size();

   _permutation.resize(sz);
   for (int ndx=0; ndx<sz; ndx++)
      _permutation[ndx] = ndx;

   for (unsigned int rounds = 0; rounds < 2; rounds++)
   {
      for (unsigned int ndx = 0; ndx < sz; ndx++)
      {
         new_ndx = urand(sz);

         temp = (*this)[new_ndx];
         (*this)[new_ndx] = (*this)[ndx];
         (*this)[ndx] = temp;

         temp_ndx = _permutation[new_ndx];
         _permutation[new_ndx] = _permutation[ndx];
         _permutation[ndx] = temp_ndx;
      }
   }
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_DATASET_H_ */
