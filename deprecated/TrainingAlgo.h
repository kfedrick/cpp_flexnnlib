//
// Created by kfedrick on 9/1/19.
//

#ifndef FLEX_NEURALNET_TRAININGALGO_H_
#define FLEX_NEURALNET_TRAININGALGO_H_

#include "EnumeratedDataSet.h"
#include "TrainingRecord.h"
#include "TrainerConfig.h"

using flexnnet::TrainerConfig;

namespace flexnnet
{
   enum EpochStatus
   {
      UPDATED = 1, FAILBACK = 0
   };

   template<class _NNIn, class _NNOut, template<class, class> class _SampleType>
   class TrainingAlgo : public TrainerConfig
   {
   public:

      void initialize_network(void);

   protected:
      EpochStatus
      train_epoch(NeuralNet <_NNIn, _NNOut>& _nnet, const ExemplarSet <_NNIn, _NNOut, _SampleType>& _trnset, const ExemplarSet <_NNIn, _NNOut, _SampleType>& _vldset, const ExemplarSet <_NNIn, _NNOut, _SampleType>& _tstset);
      void train_sample(NeuralNet <_NNIn, _NNOut>& _nnet, const _SampleType<_NNIn, _NNOut>& _asample);

   };

   template<class _NNIn, class _NNOut, template<class, class> class _SampleType>
   void TrainingAlgo<_NNIn, _NNOut, _SampleType>::initialize_network(void)
   {
      // TODO - implement
   }

   template<class _NNIn, class _NNOut, template<class, class> class _SampleType>
   EpochStatus TrainingAlgo<_NNIn,
                            _NNOut,
                            _SampleType>::train_epoch(NeuralNet <_NNIn, _NNOut>& _nnet, const ExemplarSet <_NNIn, _NNOut, _SampleType>& _trnset, const ExemplarSet <_NNIn, _NNOut, _SampleType>& _vldset, const ExemplarSet <_NNIn, _NNOut, _SampleType>& _tstset)
   {
      std::cout << "TrainAlgo::train_epoch() - entry\n";

      // Iterate through all samples in the training set_weights
      int i = 0;
      for (auto asample : _trnset)
      {
         std::cout << "TrainAlgo::train_run() - sample " << i << "\n";
         train_sample(_nnet, asample);
         i++;
      }
   }

   template<class _NNIn, class _NNOut, template<class, class> class _SampleType>
   void
   TrainingAlgo<_NNIn, _NNOut, _SampleType>::train_sample(NeuralNet <_NNIn, _NNOut>& _nnet, const _SampleType<_NNIn,
                                                                                                              _NNOut>& _asample)
   {
      std::cout << "TrainAlgo::train_sample() - entry\n";
   }
}

#endif //FLEX_NEURALNET_TRAININGALGO_H_
