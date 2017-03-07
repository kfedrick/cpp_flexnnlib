/*
 * Pattern.h
 *
 *  Created on: Feb 4, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_PATTERN_H_
#define FLEX_NEURALNET_PATTERN_H_

#include <vector>
#include <fstream>

using namespace std;

namespace flex_neuralnet
{

class Pattern
{
public:
   Pattern(int sz = 0);
   Pattern(const vector<unsigned int>& vsizes);
   Pattern(const vector< vector<double> >& vec);
   Pattern(const vector<double>& vec);
   ~Pattern();

   int size() const;

   const vector<double>& operator()(void) const;
   const vector<double>& at(int index) const;
   const vector<double>& operator[](int index) const;

   void operator=(const Pattern& patt);
   void operator=(const vector< vector<double> >& vec);
   void operator=(const vector<double>& vec);

   void push_back(const vector<double>& vec);

   void clear();

   operator const vector<double>&() const;

   void toFile(fstream& _fs);
   void fromFile(fstream& _fs);

private:
   void coalesce() const;

private:
   vector< vector<double> > data;
   mutable vector<double> coalesced_data;

   // Flag to indicate that the DataVectors have been coalesced
   mutable bool coalesced_flag;
};

}

#endif /* FLEX_NEURALNET_PATTERN_H_ */
