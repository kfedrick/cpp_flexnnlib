//
// Created by kfedrick on 9/9/19.
//

#ifndef FLEX_NEURALNET_PERFORMANCEMETRICS_H_
#define FLEX_NEURALNET_PERFORMANCEMETRICS_H_

namespace flexnnet
{
   class PerformanceMetrics
   {
      /**
       * Overall performance score.
       *
       * @return - The overall performance score.
       */
      virtual double score(void)
      {};
   };
}

#endif //FLEX_NEURALNET_PERFORMANCEMETRICS_H_
