//
// Created by kfedrick on 9/29/19.
//

#ifndef FLEX_NEURALNET_EVALUATORCONFIG_H_
#define FLEX_NEURALNET_EVALUATORCONFIG_H_

#include <cstddef>

namespace flexnnet
{
   class EvaluatorConfig
   {
   public:
      /**
       * Set the number of subsamples to draw from the test data set
       * while evaluating performance.
       *
       * @param _count
       */
      void set_sampling_count(size_t _count);

      /**
       * Set the size of the subsampling to draw
       * @param _size
       */
      void set_sampling_size(size_t _size);

      size_t sampling_size(void) const;

      size_t sampling_count(void) const;


   private:
      size_t num_samplings;
      size_t sampling_sz;
   };

   inline void EvaluatorConfig::set_sampling_count(size_t _count)
   {
      num_samplings = _count;
   }

   inline void EvaluatorConfig::set_sampling_size(size_t _size)
   {
      sampling_sz = _size;
   }

   inline size_t EvaluatorConfig::sampling_count(void) const
   {
      return num_samplings;
   }

   inline size_t EvaluatorConfig::sampling_size(void) const
   {
      return sampling_sz;
   }
}

#endif //FLEX_NEURALNET_EVALUATORCONFIG_H_
