//
// Created by kfedrick on 10/2/19.
//

#ifndef FLEX_NEURALNET_ACTORCRITICTRAINER_H_
#define FLEX_NEURALNET_ACTORCRITICTRAINER_H_

#include <cstddef>

#include "TDTrainerConfig.h"
#include "evaluators/include/TDEvaluatorConfig.h"
#include "EnumeratedDataSet.h"
#include "NeuralNet.h"

namespace flexnnet
{

   template<TDForecastMode _MODE>
   class ActorCriticTrainer : public TDTrainerConfig<_MODE>
   {
      /**
       * Train neural network for up to the maximum specified epochs or until the
       * convergence criteria is met.
       *
       * @param _nnet
       * @param _trnset
       * @param _tstset
       * @return
       */
      template<class _NNIn, class _NNOut, template<class __NNIn, class __NNOut> class _NN,
         template<class _SampleIn, class _SampleOut> class _Sample>
      void train(NeuralNet<_NNIn, _NNOut>& _nnet, const EnumeratedDataSet<_NNIn, _NNOut>& _trnset, const EnumeratedDataSet<_NNIn,
                                                                                                                           _NNOut>& _tstset = EnumeratedDataSet<
         _NNIn,
         _NNOut>());

   };
}

#endif //FLEX_NEURALNET_ACTORCRITICTRAINER_H_
