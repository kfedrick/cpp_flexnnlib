//
// Created by kfedrick on 3/12/22.
//

#ifndef FLEX_NEURALNET_BASEEVALUATOR_H_
#define FLEX_NEURALNET_BASEEVALUATOR_H_


#include <flexnnet.h>
#include <Exemplar.h>
#include <BasicEvalConfig.h>
#include "Reinforcement.h"

namespace flexnnet
{
   template<class InTyp, class OutTyp, template<class, class>
      class SampleTyp,
      template<class, class> class NN,
      template<class, class, template<class, class> class> class DataSet,
      template<class> class FitnessFunc>
   class BaseEvaluator : public BasicEvalConfig, public FitnessFunc<OutTyp>
   {
      using NNTyp = NN<InTyp, OutTyp>;
      using DatasetTyp = DataSet<InTyp, OutTyp, SampleTyp>;

   public:
      BaseEvaluator();

      EvalResults
      evaluate(NNTyp& _nnet, const DatasetTyp& _tstset);

   protected:
      double
      evaluate_subsampling(size_t _s_index, NNTyp& _nnet, const DatasetTyp& _tstset);

      virtual void
      evaluate_sample(size_t _s_index, NNTyp& _nnet, const SampleTyp<InTyp, OutTyp>& _sample) = 0;
   };

   template<class InTyp, class OutTyp, template<class, class>
      class SampleTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   BaseEvaluator<InTyp,
             OutTyp,
             SampleTyp,
             NN,
             Dataset,
             FitnessFunc>::BaseEvaluator() : FitnessFunc<OutTyp>()
   {
   }


   template<class InTyp, class OutTyp, template<class, class>
      class SampleTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   std::tuple<double, double>
   BaseEvaluator<InTyp,
             OutTyp,
             SampleTyp,
             NN,
             Dataset,
             FitnessFunc>::evaluate(NNTyp& _nnet, const DatasetTyp& _tstset)
   {
      std::cout << "BaseEvaluator::evaluate()\n";

      // Vector to hold performance results for each sampling
      size_t scount = sampling_count();
      std::valarray<double> perf(scount);
      FitnessFunc<OutTyp>::clear();

      // Iterate through all exemplars in the training set_weights
      for (size_t sample_ndx = 0; sample_ndx < scount; sample_ndx++)
      {
         if (!randomize_order())
            _tstset.normalize_order();

         perf[sample_ndx] = evaluate_subsampling(sample_ndx, _nnet, _tstset);
      }

      // Calculate performance mean and standard deviation across samples
      double sample_mean = perf.sum() / scount;
      double stdev = 0;
      for (size_t i = 0; i < scount; i++)
         stdev += (sample_mean - perf[i]) * (sample_mean - perf[i]);

      double svar = (scount > 30) ? stdev / (scount - 1) : stdev / scount;
      stdev = sqrt(svar);

      return EvalResults(sample_mean, stdev);
   }

   template<class InTyp, class OutTyp, template<class, class>
      class SampleTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   double
   BaseEvaluator<InTyp,
             OutTyp,
             SampleTyp,
             NN,
             Dataset,
             FitnessFunc>::evaluate_subsampling(size_t _sample_ndx, NNTyp& _nnet, const DatasetTyp& _tstset)
   {
      FitnessFunc<OutTyp>::clear();

      if (randomize_order())
         _tstset.randomize_order();

      size_t subsample_sz = subsample_fraction() * _tstset.size();
      size_t sample_no = 0;
      for (const SampleTyp<InTyp,OutTyp>& it : _tstset)
      {
         if (sample_no >= subsample_sz)
            break;

         evaluate_sample(sample_no++, _nnet, it);
      }

      return FitnessFunc<OutTyp>::calc_fitness();
   }
}


#endif // FLEX_NEURALNET_BASEEVALUATOR_H_
