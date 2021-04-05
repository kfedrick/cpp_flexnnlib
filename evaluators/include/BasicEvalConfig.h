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
      static constexpr size_t DEFAULT_SAMPLING_COUNT = 1;
      static constexpr size_t DEFAULT_SUBSAMPLE_FRACTION = 1.0;

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
      void set_subsample_fraction(double _frac);

      size_t sampling_count(void) const;

      double subsample_fraction(void) const;

      void randomize_order(bool _flag);

      bool randomize_order(void) const;

   private:
      bool randomize_order_flag{false};
      size_t num_samplings{DEFAULT_SAMPLING_COUNT};
      double sub_sample_fraction{DEFAULT_SUBSAMPLE_FRACTION};
   };

   inline bool BasicEvalConfig::randomize_order(void) const
   {
      return randomize_order_flag;
   }

   inline void BasicEvalConfig::randomize_order(bool _flag)
   {
      randomize_order_flag = _flag;
   }

   inline void BasicEvalConfig::set_sampling_count(size_t _count)
   {
      num_samplings = _count;
   }

   inline void BasicEvalConfig::set_subsample_fraction(double _frac)
   {
      if (_frac <= 0 || _frac > 1.0)
      {
         static std::stringstream sout;
         sout << "Error : BasicEvalConfig::set_subsample_fraction() - "
              << "Bad value, must be 0 < _frac <= 1 : \"" << _frac << "\"\n";
         throw std::invalid_argument(sout.str());
      }
      sub_sample_fraction = _frac;
   }

   inline size_t BasicEvalConfig::sampling_count(void) const
   {
      return num_samplings;
   }

   inline double BasicEvalConfig::subsample_fraction(void) const
   {
      return sub_sample_fraction;
   }
}

#endif //FLEX_NEURALNET_BASICEVALCONFIG_H_
