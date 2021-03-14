//
// Created by kfedrick on 3/3/21.
//

#ifndef FLEX_NEURALNET_EVALUATOR_H_
#define FLEX_NEURALNET_EVALUATOR_H_

#include "BasicEvalConfig.h"

namespace flexnnet
{
   template<class _InTyp, class _OutTyp,
      template<class, class> class _NN,
      template<class, class> class _DataSet,
      template<class> class _FitnessFunc>
   class Evaluator : public BasicEvalConfig, public _FitnessFunc<_OutTyp>
   {
      using _NNTyp = _NN<_InTyp, _OutTyp>;
      using _DatasetTyp = _DataSet<_InTyp, _OutTyp>;
      using _Exemplar = std::pair<_InTyp, _OutTyp>;

   public:
      Evaluator(const _DatasetTyp& _tstset);

      std::tuple<double, double>
      evaluate(_NNTyp& _nnet, const _DatasetTyp& _tstset);

   protected:
      double
      evaluate_subsampling(size_t _s_index, _NNTyp& _nnet, const _DatasetTyp& _tstset);

      void
      evaluate_sample(size_t _s_index, _NNTyp& _nnet, const _Exemplar& _sample);
   };

   template<class _In, class _Out, template<class, class> class _NN,
      template<class, class> class _Dataset,
      template<class> class _FitnessFunc>
   Evaluator<_In,
             _Out,
             _NN,
             _Dataset,
             _FitnessFunc>::Evaluator(const _DatasetTyp& _tstset)
   {
      std::cout << "Evaluator::Evaluator()\n";
   }

   template<class _In, class _Out, template<class, class> class _NN,
      template<class, class> class _Dataset,
      template<class> class _FitnessFunc>
   std::tuple<double, double>
   Evaluator<_In,
             _Out,
             _NN,
             _Dataset,
             _FitnessFunc>::evaluate(_NNTyp& _nnet, const _DatasetTyp& _tstset)
   {
      std::cout << "Evaluator::evaluate()\n";

      // Vector to hold performance results for each sampling
      size_t scount = sampling_count();
      std::valarray<double> perf(scount);

      // Iterate through all exemplars in the training set_weights
      for (size_t sample_ndx = 0; sample_ndx < scount; sample_ndx++)
      {
         // TODO - generate sampling mask and pass to evaluate_subsampling
         perf[sample_ndx] = evaluate_subsampling(sample_ndx, _nnet, _tstset);
      }

      // Calculate performance mean and standard deviation across samples
      double sample_mean = perf.sum() / scount;
      double stdev = 0;
      for (size_t i = 0; i < scount; i++)
         stdev += (sample_mean - perf[i]) * (sample_mean - perf[i]);
      stdev = sqrt(stdev / (scount - 1));
      return std::tuple<double, double>(sample_mean, stdev);
   }

   template<class _In, class _Out, template<class, class> class _NN,
      template<class, class> class _Dataset,
      template<class> class _FitnessFunc>
   double
   Evaluator<_In,
             _Out,
             _NN,
             _Dataset,
             _FitnessFunc>::evaluate_subsampling(size_t _sample_ndx, _NNTyp& _nnet, const _DatasetTyp& _tstset)
   {
      std::cout << "Evaluator::evaluate_subsampling(<< " << _sample_ndx << ")\n";
      size_t sample_no = 0;
      for (auto& it : _tstset)
         evaluate_sample(sample_no++, _nnet, it);
   }

   template<class _In, class _Out, template<class, class> class _NN,
      template<class, class> class _Dataset,
      template<class> class _FitnessFunc>
   void
   Evaluator<_In,
             _Out,
             _NN,
             _Dataset,
             _FitnessFunc>::evaluate_sample(size_t _s_index, _NNTyp& _nnet, const _Exemplar& _sample)
   {
      std::cout << "Evaluator::evaluate_sample(<< " << _s_index << ")\n" << std::flush;

      _nnet.activate(_sample.first);
      _FitnessFunc<_Out>::analyze_sample(_sample.second, _sample.second);
   }

}

#endif //FLEX_NEURALNET_EVALUATOR_H_
