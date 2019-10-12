/*
 * Pattern.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: kfedrick
 */

#include <iostream>
#include "Pattern.h"

namespace flexnnet
{

   Pattern::Pattern(int sz)
   {
      this->data.resize(sz);
      coalesced_flag = false;
   }

   Pattern::Pattern(const std::vector<unsigned int>& vsizes)
   {
      this->data.resize(vsizes.size());
      for (unsigned int ndx = 0; ndx < vsizes.size(); ndx++)
         data[ndx] = std::vector<double>(vsizes[ndx]);

      coalesced_flag = false;
   }

   Pattern::Pattern(const std::vector<double>& vec)
   {
      this->data.push_back(vec);
      coalesced_flag = false;
   }

   Pattern::Pattern(const std::vector<std::vector<double> >& vec)
   {
      this->data = vec;
      coalesced_flag = false;
   }

   Pattern::~Pattern()
   {
//   cout << "Pattern::~Pattern()" << endl;
   }

   int Pattern::size() const
   {
      return data.size();
   }

   const std::vector<double>& Pattern::operator()(void) const
   {
      coalesce();
      return coalesced_data;
   }

   const std::vector<double>& Pattern::at(int index) const
   {
      return data.at(index);
   }

   const std::vector<double>& Pattern::operator[](int index) const
   {
      return data[index];
   }

   void Pattern::operator=(const Pattern& patt)
   {
      this->data = patt.data;
      coalesced_flag = false;
   }

   void Pattern::operator=(const std::vector<std::vector<double> >& vec)
   {
      this->data = vec;
      coalesced_flag = false;
   }

   void Pattern::operator=(const std::vector<double>& vec)
   {
      this->data.clear();
      this->data.push_back(vec);
      coalesced_flag = false;
   }

   void Pattern::push_back(const std::vector<double>& vec)
   {
      this->data.push_back(vec);
   }

   void Pattern::clear()
   {
      this->data.clear();
      coalesced_flag = false;
   }

   Pattern::operator const std::vector<double>&() const
   {
      coalesce();
      return coalesced_data;
   }

   void Pattern::toFile(std::fstream& _fs)
   {
      unsigned int num_patterns = size();
      _fs.write((char*) &num_patterns, sizeof(unsigned int));

      for (unsigned int pndx = 0; pndx < num_patterns; pndx++)
      {
         const std::vector<double>& pvec = data.at(pndx);

         unsigned int pvec_size = pvec.size();
         _fs.write((char*) &pvec_size, sizeof(unsigned int));

         for (unsigned int vndx = 0; vndx < pvec_size; vndx++)
            _fs.write((char*) &pvec[vndx], sizeof(double));
      }
   }

   void Pattern::fromFile(std::fstream& _fs)
   {
      data.clear();

      unsigned int num_patterns = size();
      _fs.read((char*) &num_patterns, sizeof(unsigned int));

      std::vector<double> pvec;
      for (unsigned int pndx = 0; pndx < num_patterns; pndx++)
      {
         unsigned int pvec_size = pvec.size();
         _fs.read((char*) &pvec_size, sizeof(unsigned int));

         pvec.resize(pvec_size);
         for (unsigned int vndx = 0; vndx < pvec_size; vndx++)
            _fs.read((char*) &pvec[vndx], sizeof(double));

         data.push_back(pvec);
      }

      coalesced_flag = false;
   }

   void Pattern::coalesce() const
   {
      if (coalesced_flag)
         return;

      coalesced_data.clear();

      unsigned int i, j;
      unsigned int coalesced_size = 0;
      for (i = 0; i < data.size(); i++)
         coalesced_size += data[i].size();

      coalesced_data.resize(coalesced_size);

      unsigned int coalesced_ndx = 0;
      for (i = 0; i < data.size(); i++)
      {
         const std::vector<double>& vec = data[i];
         for (int j = 0; j < data[i].size(); j++)
            coalesced_data[coalesced_ndx++] = data[i][j];
      }

      coalesced_flag = true;
   }

} /* namespace flexnnet */
