//
// Created by kfedrick on 9/29/19.
//
// Basic neural network evaluator configuration paramaters.
//
//

#ifndef FLEX_NEURALNET_BASICEVALCONFIG_H_
#define FLEX_NEURALNET_BASICEVALCONFIG_H_

#include <cstddef>

namespace flexnnet
{
   /**
    * Basic neural network evaluator configuration parameters.
    *
    * sampling_count - the number of samplings to draw from the test data set
    *    for evaluation. The evaluator will report the mean and standard deviation
    *    of the evaluation criteria.
    *
    * sample_size - the sample size to draw from the test data set for each sampling.
    *    If the sample size is greater than the data set size the behavior depends
    *    on the specific implementation - some implementations may allow oversampling.
    */
   class BasicEvalConfig
   {
   public:
      static constexpr size_t DEFAULT_SAMPLE_SIZE = 0;
      static constexpr size_t DEFAULT_SAMPLING_COUNT = 1;

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
      void set_sample_size(size_t _size);

      size_t sample_size(void) const;

      size_t sampling_count(void) const;


   private:
      size_t num_samplings {DEFAULT_SAMPLING_COUNT};
      size_t sampling_sz {DEFAULT_SAMPLE_SIZE};
   };

   inline void BasicEvalConfig::set_sampling_count(size_t _count)
   {
      num_samplings = _count;
   }

   inline void BasicEvalConfig::set_sample_size(size_t _size)
   {
      sampling_sz = _size;
   }

   inline size_t BasicEvalConfig::sampling_count(void) const
   {
      return num_samplings;
   }

   inline size_t BasicEvalConfig::sample_size(void) const
   {
      return sampling_sz;
   }
}

#endif //FLEX_NEURALNET_BASICEVALCONFIG_H_
