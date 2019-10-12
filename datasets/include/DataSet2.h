//
// Created by kfedrick on 10/8/19.
//

#ifndef FLEX_NEURALNET_DATASET2_H_
#define FLEX_NEURALNET_DATASET2_H_

#include <set>

namespace flexnnet
{
   template<typename _Sample>
   class DataSet2 : public std::set<_Sample>
   {
      void initialize(void)
      {};
      void randomize_order(void)
      {};
   };
}

#endif //FLEX_NEURALNET_DATASET2_H_
