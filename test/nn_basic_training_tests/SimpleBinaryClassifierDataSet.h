//
// Created by kfedrick on 3/14/21.
//

#ifndef _SIMPLEBINARYCLASSIFIERDATASET_H_
#define _SIMPLEBINARYCLASSIFIERDATASET_H_

#include <DataSet.h>
#include <ValarrayMap.h>

using flexnnet::DataSet;
using flexnnet::ValarrayMap;

class SimpleBinaryClassifierDataSet : public DataSet<ValarrayMap, ValarrayMap>
{
public:
   SimpleBinaryClassifierDataSet();

   double urand() const;
   double urand(double a, double b) const;
   double nrand(double mean, double stdev);

   void generate_samples(unsigned int _num, unsigned int _class, double _mean, double _stdev);
};

inline
double SimpleBinaryClassifierDataSet::urand() const
{
   return rand() / double(RAND_MAX);
}

inline
double SimpleBinaryClassifierDataSet::urand(double a, double b) const
{
   return (b-a)*urand() + a;
}

inline
double SimpleBinaryClassifierDataSet::nrand(double mean, double stdev)
{
   double pi = 3.1415926535897;
   double r1, r2;
   r1 = urand();
   r2 = urand();
   return mean + stdev * sqrt(-2 * log(r1)) * cos(2 * pi * r2);
}

using flexnnet::Exemplar;

inline
SimpleBinaryClassifierDataSet::SimpleBinaryClassifierDataSet()
{
   srand (time(NULL));
}

inline
void SimpleBinaryClassifierDataSet::generate_samples(unsigned int _num, unsigned int _class, double _mean, double _stdev)
{
   double val;
   ValarrayMap inmap, tgtmap;

   if (_class > 1)
      std::cout << "Error: Binary classifier must have class of [0,1]\n";

   inmap["input"] = std::valarray<double>(1);
   tgtmap["output"] = std::valarray<double>(1);

   flexnnet::Exemplar<ValarrayMap, ValarrayMap> exemplar;

   for (size_t i=0; i<_num; i++)
   {
      val = nrand(_mean, _stdev);
      inmap["input"][0] = val;
      tgtmap["output"][0] = (_class == 0) ? 1.0 : -1.0;

      push_back(Exemplar<ValarrayMap,ValarrayMap>(inmap,tgtmap));
   }
}

#endif //_SIMPLEBINARYCLASSIFIERDATASET_H_
