//
// Created by kfedrick on 3/14/21.
//

#ifndef _BOUNDEDRANDOMWALKDATASET_H_
#define _BOUNDEDRANDOMWALKDATASET_H_

#include <DataSet.h>
#include <ExemplarSeries.h>

using flexnnet::DataSet;
using flexnnet::FeatureSetImpl;
using flexnnet::Exemplar;
using flexnnet::ExemplarSeries;

template<size_t N>
class BoundedRandomWalkDataSet : public DataSet<FeatureSetImpl<std::tuple<RawFeature<9 + 2>>>,
                                                FeatureSetImpl<std::tuple<RawFeature<1>>>,
                                                ExemplarSeries>
{
public:
   BoundedRandomWalkDataSet();

   /**
    *
    * @param _num     Number of samples to generate
    * @param _size    Number of non-terminal of random walk states
    * @param _rprop   Probability of a move to right
    */
   void generate_final_cost_samples(unsigned int _num, double _rprop = 0.5);

   /**
    *
    * @param _num     Number of samples to generate
    * @param _size    Number of non-terminal of random walk states
    * @param _rprop   Probability of a move to right
    */
   void generate_cost_to_go_samples(unsigned int _num, double _rprop = 0.5);

private:
   mutable std::mt19937_64 rand_engine;
};

template<size_t N> BoundedRandomWalkDataSet<N>::BoundedRandomWalkDataSet()
{
   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);
}

template<size_t N>
void BoundedRandomWalkDataSet<N>::generate_final_cost_samples(unsigned int _num, double _rprop)
{
   double val;

   FeatureSetImpl<std::tuple<RawFeature<N + 2>>> inmap({"input"});
   FeatureSetImpl<std::tuple<RawFeature<1>>> tgtmap({"output"});

   std::valarray<double> invec(N + 2);
   std::valarray<double> tgtvec = {0.0};

   flexnnet::Exemplar<FeatureSetImpl<std::tuple<RawFeature<N + 2>>>,
                      FeatureSetImpl<std::tuple<RawFeature<1>>>> exemplar;
   flexnnet::ExemplarSeries<FeatureSetImpl<std::tuple<RawFeature<N + 2>>>,
                            FeatureSetImpl<std::tuple<RawFeature<1>>>> series;

   std::uniform_int_distribution<int> binary_dist(1, 10000);
   std::uniform_int_distribution<int> uniform_dist(1, N);
   //std::uniform_int_distribution<int> uniform_dist(4, 6);

   //clear();
   for (size_t i = 0; i < _num; i++)
   {
      series.clear();

      // Get starting location
      int position_ndx = uniform_dist(rand_engine);
      invec = -1.0;
      invec[position_ndx] = 1.0;

      do
      {
         std::get<0>(inmap.get_features()).decode(invec);
         std::get<0>(tgtmap.get_features()).decode(tgtvec);
         series.push_back(Exemplar<FeatureSetImpl<std::tuple<RawFeature<N + 2>>>,
                                   FeatureSetImpl<std::tuple<RawFeature<1>>>>(inmap, tgtmap,
                                                                              false));

         position_ndx += (binary_dist(rand_engine) <= 5000) ? -1 : 1;
         invec = -1.0;
         invec[position_ndx] = 1.0;

      }
      while (0 < position_ndx && position_ndx < N + 1);

      tgtvec[0] = {(position_ndx == 0) ? -1.0 : 1.0};

      std::get<0>(inmap.get_features()).decode(invec);
      std::get<0>(tgtmap.get_features()).decode(tgtvec);
      series.push_back(Exemplar<FeatureSetImpl<std::tuple<RawFeature<N + 2>>>,
                                FeatureSetImpl<std::tuple<RawFeature<1>>>>(inmap, tgtmap));

      push_back(series);
   }
}

template<size_t N>
void BoundedRandomWalkDataSet<N>::generate_cost_to_go_samples(unsigned int _num, double _rprop)
{
   double val;
   FeatureSetImpl<std::tuple<RawFeature<N + 2>>> inmap({"input"});
   FeatureSetImpl<std::tuple<RawFeature<1>>> tgtmap({"output"});

   std::valarray<double> invec(N + 2);
   std::valarray<double> tgtvec = {1.0};

   flexnnet::Exemplar<FeatureSetImpl<std::tuple<RawFeature<N + 2>>>,
                      FeatureSetImpl<std::tuple<RawFeature<1>>>> exemplar;
   flexnnet::ExemplarSeries<FeatureSetImpl<std::tuple<RawFeature<N + 2>>>,
                            FeatureSetImpl<std::tuple<RawFeature<1>>>> series;

   std::uniform_int_distribution<int> binary_dist(1, 10000);
   std::uniform_int_distribution<int> uniform_dist(1, N);

   //clear();
   for (size_t i = 0; i < _num; i++)
   {
      series.clear();

      // Get starting location
      int position_ndx = uniform_dist(rand_engine);
      invec = -1.0;
      invec[position_ndx] = 1.0;

      tgtvec[0] = 1.0;

      do
      {
         std::get<0>(inmap.get_features()).decode(invec);
         std::get<0>(tgtmap.get_features()).decode(tgtvec);
         series.push_back(Exemplar<FeatureSetImpl<std::tuple<RawFeature<N + 2>>>,
                                   FeatureSetImpl<std::tuple<RawFeature<1>>>>(inmap, tgtmap));

         position_ndx += (binary_dist(rand_engine) < 5000) ? -1 : 1;
         invec = -1.0;
         invec[position_ndx] = 1.0;
      }
      while (0 < position_ndx && position_ndx < N + 1);

      tgtvec = {0.0};

      std::get<0>(inmap.get_features()).decode(invec);
      std::get<0>(tgtmap.get_features()).decode(tgtvec);
      series.push_back(Exemplar<FeatureSetImpl<std::tuple<RawFeature<N + 2>>>,
                                FeatureSetImpl<std::tuple<RawFeature<1>>>>(inmap, tgtmap));

      push_back(series);
   }
}

#endif //_BOUNDEDRANDOMWALKDATASET_H_
