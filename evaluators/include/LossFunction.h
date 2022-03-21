//
// Created by kfedrick on 3/13/22.
//

#ifndef FLEX_NEURALNET_LOSSFUNCTION_H_
#define FLEX_NEURALNET_LOSSFUNCTION_H_

#include <flexnnet.h>
#include <tuple>

namespace flexnnet
{
   template<class InTyp, class TgtTyp, template<class, class> class SampleTyp> class LossFunction
   {
      using NNTyp = NeuralNet<InTyp, TgtTyp>;
      using DatasetTyp = DataSet<InTyp, TgtTyp, SampleTyp>;

   public:
      static constexpr unsigned int DEFAULT_SUBSAMPLE_COUNT = 1;
      static constexpr double DEFAULT_FITNESS_SUBSAMPLE_SZ = 0;
      static constexpr double DEFAULT_SE_SUBSAMPLE_FRACTION = 1.0;

   protected:
      /**
       * protected: LossFunction()
       */
      LossFunction();

   public:
      /**
       * public: calc_fitness_standard_error
       *
       * @param _nnet
       * @param _tstset
       * @param _subsample_fraction : The fraction of samples in the specified
       *    data set to use randomly include in each sub-sample. Specifying
       *    a value of zero will cause a single data set entry to be used
       *    for each sub-sample.
       * @return tuple<double,double>(mean,stderr)
       *
       * @details
       *    Returns the mean and standard error of the calculated fitness
       *    values of the specified neural net on a randomly drawn set of
       *    sub-samples over the specified data set. The number of samples
       *    is set by the set_subsample_count(unsigned int) member function.
       */
      virtual std::tuple<double, double> calc_fitness_standard_error(
         NNTyp& _nnet, const DatasetTyp& _tstset,
         double _subsample_fraction = DEFAULT_SE_SUBSAMPLE_FRACTION);

      virtual double calc_fitness(
         NNTyp& _nnet, const DatasetTyp& _tstset,
         unsigned int _subsample_sz = DEFAULT_FITNESS_SUBSAMPLE_SZ) = 0;

      /**
       * public: set_subsample_count
       *
       * Set the number of sub-samples to be drawn for calculating the
       * standard error of the fitness function values.
       *
       * @param _sz
       */
      void set_subsample_count(unsigned int _sz);

      /**
       * public: get_subsample_count
       *
       * @return the current setting for the number of sub-sample to be drawn.
       */
      unsigned int get_subsample_count();

   private:
      std::valarray<double> performance_cache;

   };

   template<class InTyp, class TgtTyp, template<class, class> class SampleTyp>
   LossFunction<InTyp, TgtTyp, SampleTyp>::LossFunction()
   {
      performance_cache.resize(1);
   }

   template<class InTyp, class TgtTyp, template<class, class> class SampleTyp>
   void LossFunction<InTyp, TgtTyp, SampleTyp>::set_subsample_count(unsigned int _sz)
   {
      if (_sz == 0)
         throw std::invalid_argument("invalid subsample size : 0 (zero)");

      performance_cache.resize(_sz);
   }

   template<class InTyp, class TgtTyp, template<class, class> class SampleTyp>
   unsigned int LossFunction<InTyp, TgtTyp, SampleTyp>::get_subsample_count()
   {
      return performance_cache.size();
   }

   template<class InTyp, class TgtTyp, template<class, class> class SampleTyp>
   std::tuple<double, double> LossFunction<InTyp, TgtTyp, SampleTyp>::calc_fitness_standard_error(
      NNTyp& _nnet, const DatasetTyp& _tstset, double _subsample_fraction)
   {
      /*
       * Validate _subsample_fraction value is greater than or equal to
       * 0.0 and less than or equal to 1.0
       */
      if (_subsample_fraction <= 0.0)
      {
         static std::stringstream sout;
         sout << "Error : LossFunction::calc_fitness_standard_error - "
              << "Invalid sub-sample fraction specified : \"" << _subsample_fraction
              << "\" : must be 0 to 1.0\n";
         throw std::invalid_argument(sout.str());
      }

      unsigned int num_subsamples = get_subsample_count();

      unsigned int subsample_sz;
      subsample_sz = (_subsample_fraction > 0) ? _subsample_fraction * _tstset.size() : 1;
      if (subsample_sz == 0)
         subsample_sz = 1;

      // Iterate through all exemplars in the training set_weights
      for (size_t sample_ndx = 0; sample_ndx < num_subsamples; sample_ndx++)
      {
         _tstset.randomize_order();
         performance_cache[sample_ndx] = calc_fitness(_nnet, _tstset, subsample_sz);
      }

      // Calculate fitness mean and standard error across samples
      double sample_mean, sample_std_error;

      sample_mean = performance_cache.sum() / num_subsamples;
      double diff, sumsqrdiff = 0;
      for (size_t i = 0; i < num_subsamples; i++)
      {
         diff = sample_mean - performance_cache[i];
         sumsqrdiff += diff * diff;
      }
      double sample_var =
         (num_subsamples > 30) ? sumsqrdiff / (num_subsamples - 1) : sumsqrdiff / num_subsamples;
      sample_std_error = sqrt(sample_var);

      return std::make_tuple(sample_mean, sample_std_error);
   }
}

#endif // FLEX_NEURALNET_LOSSFUNCTION_H_
