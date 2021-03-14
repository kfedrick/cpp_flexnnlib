//
// Created by kfedrick on 9/23/19.
//

#ifndef FLEX_NEURALNET_BASICEVALUATOR_H_
#define FLEX_NEURALNET_BASICEVALUATOR_H_

#include <cstddef>
#include <iostream>

#include "EnumeratedDataSet.h"
#include "Episode.h"
#include "BasicEvalConfig.h"
#include "NeuralNet.h"
#include "RMSError.h"

namespace flexnnet
{
   template<class _NNIn, class _NNOut, template<class, class> class _Sample, template<class> class _ErrFunc = RMSError>
   class BasicEvaluator : public BasicEvalConfig, public _ErrFunc<_NNOut>
   {
   protected:
      using _index_typ = size_t;
      using NN_Typ_ = NeuralNet<_NNIn, _NNOut>;
      using DataSet_Typ_ = EnumeratedDataSet<_NNIn, _NNOut>;
      using Exemplar_Typ_ = std::tuple<_NNIn, _NNOut>;
      using Episode_Typ_ = Episode<_NNIn, _NNOut>;

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

      double evaluate_exemplar(NN_Typ_& _nnet, const Exemplar_Typ_& _exemplar);
   };

   template<class _In, class _Out, template<class, class> class _Sample, template<class> class _Err>
   double BasicEvaluator<_In, _Out, _Sample, _Err>::evaluate(NN_Typ_& _nnet, const DataSet_Typ_& _tstset)
   {
      std::cout << "FAEvaluator::evaluate()\n";

      // Vector to hold performance results for each sampling
      size_t scount = sampling_count();
      std::valarray<double> perf(scount);

      // Iterate through all exemplars in the training set_weights
      for (size_t sampling_ndx = 0; sampling_ndx < scount; sampling_ndx++)
      {
         // TODO - generate sampling mask and pass to evaluate_subsampling
         perf[sampling_ndx] = evaluate_subsampling(_nnet, _tstset);
      }

      // TODO - calculate statistics across samplings

      return perf.sum() / scount;
   }

   template<class _In, class _Out, template<class, class> class _Sample, template<class> class _Err>
   double
   BasicEvaluator<_In, _Out, _Sample, _Err>::evaluate_subsampling(NN_Typ_& _nnet, const DataSet_Typ_& _tstset)
   {
      std::cout << "FAEvaluator::evaluate_subsampling()\n";

      // Vector to hold performance results for each sampling
      std::valarray<double> perf(_tstset.size());

      // Iterate through all exemplars in the training set_weights. NOTE EnumeratedDataSet is
      // only guaranteed to be iterable so we can't iterate by index.
      _index_typ exemplar_ndx = 0;
      for (const Exemplar_Typ_& asample : _tstset)
      {
         std::cout << "FAEvaluator::evaluate_subsampling() - sample " << exemplar_ndx << "\n";
         perf[exemplar_ndx] = evaluate_exemplar(_nnet, asample);

         exemplar_ndx++;
      }
      return perf.sum() / _tstset.size();
   }

   template<class _In, class _Out, template<class, class> class _Sample, template<class> class _Err>
   double
   BasicEvaluator<_In, _Out, _Sample, _Err>::evaluate_exemplar(NN_Typ_& _nnet, const Exemplar_Typ_& _exemplar)
   {
      std::cout << "FAEvaluator::evaluate_exemplar()\n";

      const _Out& nn_out = _nnet.activate(_exemplar.input());

      // TODO - nn_out and exemplar target must be 'vectorizable' - turn into
      // valarray before passing to the error function.
      return _Err<_Out>::error(nn_out, _exemplar.target()).error;
   }

}

#endif //FLEX_NEURALNET_BASICEVALUATOR_H_
