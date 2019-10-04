//
// Created by kfedrick on 9/23/19.
//

#ifndef FLEX_NEURALNET_FUNCAPPROXEVALUATOR_H_
#define FLEX_NEURALNET_FUNCAPPROXEVALUATOR_H_

#include <cstddef>
#include <iostream>

#include "ExemplarSet.h"
#include "EvaluatorConfig.h"
#include "NeuralNet.h"

namespace flexnnet
{
   template<class _NNIn, class _NNOut, class _ErrFunc>
   class FuncApproxEvaluator : public EvaluatorConfig
   {
   protected:
      using NN_Typ_ = NeuralNet<_NNIn, _NNOut>;
      using DataSet_Typ_ = ExemplarSet<_NNIn, _NNOut>;
      using Exemplar_Typ_ = Exemplar<_NNIn, _NNOut>;

   public:

      /**
       * Evaluate the performance of the neural net over the specified number of samplings
       * of the given test set.
       *
       * @param _nnet
       * @param _tstset
       * @return
       */
      double evaluate(NN_Typ_& _nnet, const DataSet_Typ_& _tstset);

   protected:
      /**
       * Evaluate one sampling from the dataset
       *
       * @param _nnet
       * @param _tstset
       * @return
       */
      double evaluate_subsampling(NN_Typ_& _nnet, const DataSet_Typ_& _tstset);

      double evaluate_exemplar(NN_Typ_& _nnet, Exemplar_Typ_& _exemplar);
   };

   template<class _In, class _Out, class _Err>
   double FuncApproxEvaluator<_In, _Out, _Err>::evaluate(NN_Typ_& _nnet, const DataSet_Typ_& _tstset)
   {
      std::cout << "ExemplarEvaluator::evaluate()\n";

      // Vector to hold performance results for each sampling
      size_t scount = sampling_count();
      std::valarray<double> perf(scount);

      // Iterate through all exemplars in the training set
      for (size_t sampling_ndx = 0; sampling_ndx < scount; sampling_ndx++)
      {
         // TODO - generate sampling mask and pass to evaluate_subsampling
         perf[sampling_ndx] = evaluate_subsampling(_nnet, _tstset);
      }

      // TODO - calculate statistics across samplings

      return perf.sum() / scount;
   }

   template<class _In, class _Out, class _Err>
   double FuncApproxEvaluator<_In, _Out, _Err>::evaluate_subsampling(NN_Typ_& _nnet, const DataSet_Typ_& _tstset)
   {
      std::cout << "ExemplarEvaluator::evaluate_sampling() - entry\n";

      // Vector to hold performance results for each sampling
      std::valarray<double> perf(_tstset.size());

      // Iterate through all exemplars in the training set. NOTE ExemplarSet is
      // only guaranteed to be iterable so we can't iterate by index.
      _index_typ exemplar_ndx = 0;
      for (Exemplar_Typ_& asample : _tstset)
      {
         std::cout << "ExemplarEvaluator::evaluate_sampling() - sample " << exemplar_ndx << "\n";
         perf[exemplar_ndx] = evaluate_exemplar(_nnet, asample);

         exemplar_ndx++;
      }
      return perf.sum() / _tstset.size();
   }

   template<class _In, class _Out, class _Err>
   double
   FuncApproxEvaluator<_In, _Out, _Err>::evaluate_exemplar(NN_Typ_& _nnet, Exemplar_Typ_& _exemplar)
   {
      _Out& nn_out = _nnet.activate(_exemplar.input());
      return _Err::error(nn_out, _exemplar.target());
   }

}

#endif //FLEX_NEURALNET_FUNCAPPROXEVALUATOR_H_
