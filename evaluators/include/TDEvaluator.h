//
// Created by kfedrick on 3/3/21.
//

#ifndef FLEX_NEURALNET_SERIESEVALUATOR_H_
#define FLEX_NEURALNET_SERIESEVALUATOR_H_

#include <flexnnet.h>
#include <Exemplar.h>
#include <ExemplarSeries.h>
#include <BasicEvalConfig.h>
#include "Reinforcement.h"

namespace flexnnet
{
   template<class InTyp, class OutTyp,
      template<class, class> class NN,
      template<class, class, template<class, class> class> class _DataSet,
      template<class> class FitnessFunc>
   class TDEvaluator : public BasicEvalConfig, public FitnessFunc<OutTyp>
   {
      using NNTyp = NN<InTyp, OutTyp>;
      using ExemplarTyp = Exemplar<InTyp, OutTyp>;
      using ExemplarSeriesTyp = ExemplarSeries<InTyp, OutTyp>;
      using DatasetTyp = _DataSet<InTyp, OutTyp, ExemplarSeries>;

   public:
      TDEvaluator();

      EvalResults
      evaluate(NNTyp& _nnet, const DatasetTyp& _tstset);

   protected:
      double
      evaluate_subsampling(size_t _s_index, NNTyp& _nnet, const DatasetTyp& _tstset);

      void
      evaluate_series(size_t _s_index, NNTyp& _nnet, const ExemplarSeriesTyp& _series);
   };

   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   TDEvaluator<InTyp,
               OutTyp,
               NN,
               Dataset,
               FitnessFunc>::TDEvaluator() : FitnessFunc<OutTyp>()
   {
   }


   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   std::tuple<double, double>
   TDEvaluator<InTyp,
               OutTyp,
               NN,
               Dataset,
               FitnessFunc>::evaluate(NNTyp& _nnet, const DatasetTyp& _tstset)
   {
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

   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   double
   TDEvaluator<InTyp,
               OutTyp,
               NN,
               Dataset,
               FitnessFunc>::evaluate_subsampling(size_t _sample_ndx, NNTyp& _nnet, const DatasetTyp& _tstset)
   {
      FitnessFunc<OutTyp>::clear();

      if (randomize_order())
         _tstset.randomize_order();

      size_t subsample_sz = subsample_fraction() * _tstset.size();
      size_t sample_no = 0;
      for (const ExemplarSeriesTyp& it : _tstset)
      {
         if (sample_no >= subsample_sz)
            break;

         evaluate_series(sample_no++, _nnet, it);
         FitnessFunc<OutTyp>::new_series();
      }

      return FitnessFunc<OutTyp>::calc_fitness();
   }

   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   void
   TDEvaluator<InTyp,
               OutTyp,
               NN,
               Dataset,
               FitnessFunc>::evaluate_series(size_t _s_index, NNTyp& _nnet, const ExemplarSeriesTyp& _series)
   {
      std::valarray<double> Re0, Re1;

      const ExemplarTyp& exemplar = _series[0];
      _nnet.activate(exemplar.first);

      int series_len = _series.size();
      for (int ndx=1; ndx < series_len-1; ndx++)
      {
         const OutTyp nnout0 = _nnet.vectorize_features();

         const ExemplarTyp& exemplar = _series[ndx];
         const OutTyp& nnout1 = _nnet.activate(exemplar.first);

         OutTyp tgt = FitnessFunc<OutTyp>::calc_target(exemplar.second, nnout1);
         FitnessFunc<OutTyp>::calc_error_gradient(tgt, nnout0);
      }
      const ExemplarTyp& last_exemplar = _series[series_len-1];
      const OutTyp nnout0 = _nnet.vectorize_features();

      OutTyp zeronnout;
      ValarrMap vm({{"output",{0}}});
      zeronnout.activate(vm);

      OutTyp tgt = FitnessFunc<OutTyp>::calc_target(last_exemplar.second, zeronnout);
      FitnessFunc<OutTyp>::calc_error_gradient(tgt, nnout0);
   }
}

#endif //FLEX_NEURALNET_SERIESEVALUATOR_H_
