//
// Created by kfedrick on 3/3/21.
//

#ifndef FLEX_NEURALNET_EVALUATOR_H_
#define FLEX_NEURALNET_EVALUATOR_H_

#include <flexnnet.h>
#include <Exemplar.h>
#include <BasicEvalConfig.h>

namespace flexnnet
{
   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class, template<class,class> class> class _DataSet,
      template<class> class _FitnessFunc>
   class Evaluator : public BasicEvalConfig, public _FitnessFunc<_OutTyp>
   {
      using _NNTyp = _NN<_InTyp, _OutTyp>;
      using _ExemplarTyp = Exemplar<_InTyp, _OutTyp>;
      using _DatasetTyp = _DataSet<_InTyp, _OutTyp, Exemplar>;

   public:
      Evaluator();

      EvalResults
      evaluate(_NNTyp& _nnet, const _DatasetTyp& _tstset);

   protected:
      double
      evaluate_subsampling(size_t _s_index, _NNTyp& _nnet, const _DatasetTyp& _tstset);

      void
      evaluate_sample(size_t _s_index, _NNTyp& _nnet, const _ExemplarTyp& _sample);
   };

   template<class _In, class _Out, template<class, class> class _NN,
      template<class, class, template<class,class> class> class _Dataset,
      template<class> class _FitnessFunc>
   Evaluator<_In,
             _Out,
             _NN,
             _Dataset,
             _FitnessFunc>::Evaluator()
   {
   }

   template<class _In, class _Out, template<class, class> class _NN,
      template<class, class, template<class,class> class> class _Dataset,
      template<class> class _FitnessFunc>
   std::tuple<double, double>
   Evaluator<_In,
             _Out,
             _NN,
             _Dataset,
             _FitnessFunc>::evaluate(_NNTyp& _nnet, const _DatasetTyp& _tstset)
   {
      // Vector to hold performance results for each sampling
      size_t scount = sampling_count();
      std::valarray<double> perf(scount);
      _FitnessFunc<_Out>::clear();

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

      double svar = (scount > 30) ? stdev/(scount-1) : stdev/scount;
      stdev = sqrt(svar);

      return EvalResults(sample_mean, stdev);
   }

   template<class _In, class _Out, template<class, class> class _NN,
      template<class, class, template<class,class> class> class _Dataset,
      template<class> class _FitnessFunc>
   double
   Evaluator<_In,
             _Out,
             _NN,
             _Dataset,
             _FitnessFunc>::evaluate_subsampling(size_t _sample_ndx, _NNTyp& _nnet, const _DatasetTyp& _tstset)
   {
      _FitnessFunc<_Out>::clear();

      if (randomize_order())
         _tstset.randomize_order();

      size_t subsample_sz = subsample_fraction() * _tstset.size();
      size_t sample_no = 0;
      for (auto& it : _tstset)
      {
         if (sample_no >= subsample_sz)
            break;

         evaluate_sample(sample_no++, _nnet, it);
      }

      return _FitnessFunc<_Out>::calc_fitness();
   }

   template<class _In, class _Out, template<class, class> class _NN,
      template<class, class, template<class,class> class> class _Dataset,
      template<class> class _FitnessFunc>
   void
   Evaluator<_In,
             _Out,
             _NN,
             _Dataset,
             _FitnessFunc>::evaluate_sample(size_t _s_index, _NNTyp& _nnet, const _ExemplarTyp& _sample)
   {
      const _Out& nnout = _nnet.activate(_sample.first);
      _FitnessFunc<_Out>::calc_error_gradient(_sample.second, nnout);
   }
}

#endif //FLEX_NEURALNET_EVALUATOR_H_
