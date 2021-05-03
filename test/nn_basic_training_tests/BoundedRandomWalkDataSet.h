//
// Created by kfedrick on 3/14/21.
//

#ifndef _BOUNDEDRANDOMWALKDATASET_H_
#define _BOUNDEDRANDOMWALKDATASET_H_

#include <DataSet.h>
#include <ValarrayMap.h>
#include <ExemplarSeries.h>

using flexnnet::DataSet;
using flexnnet::ValarrayMap;
using flexnnet::Exemplar;
using flexnnet::ExemplarSeries;

class BoundedRandomWalkDataSet : public DataSet<ValarrayMap, ValarrayMap, ExemplarSeries>
{
public:
   BoundedRandomWalkDataSet();

   /**
    *
    * @param _num     Number of samples to generate
    * @param _size    Number of non-terminal of random walk states
    * @param _rprop   Probability of a move to right
    */
   void generate_final_cost_samples(unsigned int _num, unsigned int _size, double _rprop = 0.5);

   /**
    *
    * @param _num     Number of samples to generate
    * @param _size    Number of non-terminal of random walk states
    * @param _rprop   Probability of a move to right
    */
   void generate_cost_to_go_samples(unsigned int _num, unsigned int _size, double _rprop = 0.5);

private:
   mutable std::mt19937_64 rand_engine;
};


inline
BoundedRandomWalkDataSet::BoundedRandomWalkDataSet()
{
   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);
}

inline
void BoundedRandomWalkDataSet::generate_final_cost_samples(unsigned int _num, unsigned int _size, double _rprop)
{
   double val;
   ValarrayMap inmap, tgtmap;

   inmap["input"] = std::valarray<double>(_size+2);
   tgtmap["output"] = std::valarray<double>(0);

   flexnnet::Exemplar<ValarrayMap, ValarrayMap> exemplar;
   flexnnet::ExemplarSeries<ValarrayMap, ValarrayMap> series;

   std::uniform_int_distribution<int> binary_dist(1, 100);
   std::uniform_int_distribution<int> uniform_dist(1, _size);

   clear();
   for (size_t i=0; i<_num; i++)
   {
      series.clear();
      tgtmap["output"].resize(0);

      // Get starting location
      int position_ndx = uniform_dist(rand_engine);
      inmap["input"] = -1.0;
      inmap["input"][position_ndx] = 1.0;

      do
      {
         series.push_back(Exemplar<ValarrayMap,ValarrayMap>(inmap, tgtmap));

         position_ndx += (binary_dist(rand_engine)<50) ? -1 : 1;
         inmap["input"] = -1.0;
         inmap["input"][position_ndx] = 1.0;

      } while (0 < position_ndx && position_ndx < _size+1);

      tgtmap["output"].resize(1);
      tgtmap["output"][0] = {(position_ndx == 0) ? -1.0 : 1.0};
      series.push_back(Exemplar<ValarrayMap,ValarrayMap>(inmap, tgtmap));

      push_back(series);
   }
}


inline
void BoundedRandomWalkDataSet::generate_cost_to_go_samples(unsigned int _num, unsigned int _size, double _rprop)
{
   double val;
   ValarrayMap inmap, tgtmap;

   inmap["input"] = std::valarray<double>(_size+2);
   tgtmap["output"] = std::valarray<double>(1);

   flexnnet::Exemplar<ValarrayMap, ValarrayMap> exemplar;
   flexnnet::ExemplarSeries<ValarrayMap, ValarrayMap> series;

   std::uniform_int_distribution<int> binary_dist(1, 100);
   std::uniform_int_distribution<int> uniform_dist(1, _size);

   clear();
   for (size_t i=0; i<_num; i++)
   {
      series.clear();

      // Get starting location
      int position_ndx = uniform_dist(rand_engine);
      inmap["input"] = -1.0;
      inmap["input"][position_ndx] = 1.0;
      tgtmap["output"][0] = {1.0};

      do
      {
         series.push_back(Exemplar<ValarrayMap,ValarrayMap>(inmap, tgtmap));

         position_ndx += (binary_dist(rand_engine)<50) ? -1 : 1;
         inmap["input"] = -1.0;
         inmap["input"][position_ndx] = 1.0;

      } while (0 < position_ndx && position_ndx < _size+1);

      tgtmap["output"][0] = {0.0};
      series.push_back(Exemplar<ValarrayMap,ValarrayMap>(inmap, tgtmap));

      push_back(series);
   }
}

#endif //_BOUNDEDRANDOMWALKDATASET_H_
