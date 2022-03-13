//
// Created by kfedrick on 3/3/21.
//

#ifndef FLEX_NEURALNET_EVALUATOR_H_
#define FLEX_NEURALNET_EVALUATOR_H_

#include <flexnnet.h>
#include <Exemplar.h>
#include <BasicEvalConfig.h>
#include "Reinforcement.h"

namespace flexnnet
{
   template<class InTyp, class OutTyp,
      template<class, class> class NN,
      template<class, class, template<class, class> class> class DataSet,
      template<class> class FitnessFunc>
   class Evaluator : public BasicEvalConfig, public FitnessFunc<OutTyp>
   {
      using NNTyp = NN<InTyp, OutTyp>;
      using _ExemplarTyp = Exemplar<InTyp,OutTyp>;
      using DatasetTyp = DataSet<InTyp, OutTyp, Exemplar>;

   public:
      Evaluator();

      EvalResults
      evaluate(NNTyp& _nnet, const DatasetTyp& _tstset);

   protected:
      double
      evaluate_subsampling(size_t _s_index, NNTyp& _nnet, const DatasetTyp& _tstset);

      void
      evaluate_exemplar(size_t _s_index, NNTyp& _nnet, const _ExemplarTyp& _exemplar);
   };

   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   Evaluator<InTyp,
             OutTyp,
             NN,
             Dataset,
             FitnessFunc>::Evaluator() : FitnessFunc<OutTyp>()
   {
   }


   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   std::tuple<double, double>
   Evaluator<InTyp,
             OutTyp,
             NN,
             Dataset,
             FitnessFunc>::evaluate(NNTyp& _nnet, const DatasetTyp& _tstset)
   {
      std::cout << "Evaluator::evaluate()\n";

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
   Evaluator<InTyp,
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
      for (const _ExemplarTyp& it : _tstset)
      {
         if (sample_no >= subsample_sz)
            break;

         evaluate_exemplar(sample_no++, _nnet, it);
      }

      return FitnessFunc<OutTyp>::calc_fitness();
   }

   template<class InTyp, class OutTyp, template<class, class> class NN,
      template<class, class, template<class, class> class> class Dataset,
      template<class> class FitnessFunc>
   void
   Evaluator<InTyp,
             OutTyp,
             NN,
             Dataset,
             FitnessFunc>::evaluate_exemplar(size_t _s_index, NNTyp& _nnet, const _ExemplarTyp& _exemplar)
   {
      std::cout << "Evaluator::evaluate_exemplar()\n";
      const OutTyp& nnout = _nnet.activate(_exemplar.first);
      FitnessFunc<OutTyp>::calc_error_gradient(_exemplar.second, nnout);
   }

}

#endif //FLEX_NEURALNET_EVALUATOR_H_
