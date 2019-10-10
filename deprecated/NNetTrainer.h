//
// Created by kfedrick on 8/21/19.
//

#ifndef FLEX_NEURALNET_BASICTRAINER_H_
#define FLEX_NEURALNET_BASICTRAINER_H_

/**
 * Given a system consisting of a 4-tuple (Neural network X training algorithm X a set of
 * training data X optimization function X evaluation function) this class evaluates the
 * performance of the neural network on the test data after training using a specified
 * training algorithm using the specified optimization function. The performance is
 * evaluated based on the specified evaluation function.
 *
 * Procedure:
 *
 * 1. For each set of training data (3-tuple training data, validation data, test data):
 * 2. Train the neural network N times using the specified optimization criteria
 *    starting from network specific initial conditions (e.g. random initial weights)
 * 3. Calculate training system statistics (e.g. percent trained to acceptance criteria, etc)
 *    according to evaluation function
 * 4. Save specified number of trained networks (e.g. M best, worst)
 */

/*
 * NOTES:
 *
 * I. Concretely defined objects:
 *    A. Neural network (more or less)
 *    B. Training set_weights (more or less)
 *    C. Optimization function (I have a current interface for this)
 *
 * II. Items yet to be concretely defined wrt interface:
 *    A. Training algorithm
 *    B. System evaluation function
 *
 * - Concrete objects that require configuration should be passed in (as constructor
 * parameters?). Policies that only apply to this class may be template classes. Not
 * necessarily. The class template can still be used even if the object is passed
 * in during construction. This would allow the evaluator to hold the specialized
 * object while still using the base interface. Why?
 *
 * Imaging SpecialNeuralNet -->> BasicNeuralNet and SpecialNeuralNet overrides some
 * BasicNeuralNet method used by Evaluator. Unless we pass the object as a pointer
 * the specialization will be lost. Using a template buys a tad bit more flexibility
 * since I can pass a reference or pointer - otherwise not much difference. On the
 * other hand it allows me to derive specializations of Evaluator depending on the
 * template type.
 *
 *
 */

#include <cstddef>
#include <memory>

#include "TrainerUtils.h"

#include "BasicNeuralNet.h"
#include "NeuralNet.h"
#include "DataSet.h"
#include "TrainingRecord.h"
#include "PerformanceMetrics.h"

namespace flexnnet
{
   template<class _NNIn, class _NNOut, template<class __NNIn, class __NNOut> class _NN,
      template<class _SampleIn, class _SampleOut> class _Sample,
      template<class __NNTIn, class _NNTOut,
      template<class _TAlgoSampleIn, class _TAlgoSampleOut> class> class _TrainAlgo>
   class NNetTrainer : public TrainerUtils, public _TrainAlgo<_NNIn,_NNOut,_Sample>
   {

   private:
      typedef _NN<_NNIn, _NNOut> NN_TYP;
      typedef _TrainAlgo<_NNIn, _NNOut, _Sample> TRAINER_TYP;
      typedef _Sample<_NNIn, _NNOut> SAMPLE_TYP;
      typedef flexnnet::DataSet<_NNIn, _NNOut, _Sample> DATASET_TYP;


   public:
      NNetTrainer (NN_TYP &_nnet, TRAINER_TYP &_trainer);

   public:
      /**
       * Train the specified number of randomly initialized neural networks
       * with the specified training, validation, and test sets and save the
       * results from the best performing networks.
       *
       * @param _trnset
       * @param _tstset
       * @param _vldset
       */
      const vector<TrainedNNetRecord>& multitrain (
         const DATASET_TYP &_trnset, const DATASET_TYP &_tstset, const DATASET_TYP &_vldset = DATASET_TYP ());
      std::shared_ptr<flexnnet::TrainingRecord> train(NeuralNet<_NNIn,_NNOut>& _nnet, const DataSet<_NNIn, _NNOut, _Sample>& _trnset, const DataSet<_NNIn, _NNOut, _Sample>& _tstset, const DataSet<_NNIn, _NNOut, _Sample>& _vldset);

   private:
      _NN<_NNIn, _NNOut> &neural_net;
      _TrainAlgo<_NNIn, _NNOut, _Sample> &train_algo;
   };

   template<class _In, class _Out, template<class _NNIn, class _NNOut> class _NN,
      template<class, class> class _Sample,
      template<class, class, template<class, class> class _TrainerSample> class _Trainer>
   NNetTrainer<
      _In,
      _Out,
      _NN,
      _Sample,
      _Trainer>::NNetTrainer (NN_TYP &_nnet, TRAINER_TYP &_trainer)
      :
      TrainerUtils(), neural_net (_nnet), train_algo (_trainer)
   {
      std::cout << "BasicTrainer::BasicTrainer() - entry\n";
   }



   template<class _In, class _Out, template<class _NNIn, class _NNOut> class _NN,
      template<class, class> class _Sample,
      template<class, class, template<class, class> class _TrainerSample> class _TrainAlgo>
   const vector<TrainedNNetRecord>& NNetTrainer<
      _In,
      _Out,
      _NN,
      _Sample,
      _TrainAlgo>::multitrain (const DATASET_TYP &_trnset, const DATASET_TYP &_tstset, const DATASET_TYP &_vldset)
   {
      std::cout << "BasicTrainer::train_run() - entry\n";

      // Perform the number of training runs specified by 'training_runs'
      for (
         unsigned int trun_ndx = 0;
         trun_ndx < const_training_runs;
         trun_ndx++)
      {
         std::cout << "BasicTrainer::train_run() - training run " << trun_ndx << "\n";

         // Initialize and train_run the network
         neural_net.initialize_weights ();
         std::shared_ptr<TrainingRecord> tr = train (neural_net, _trnset, _tstset, _vldset);

         // Collect the aggregate statistics over all training runs
         collect_training_stats (*tr);

         // Save the network if it has one of the best performances on the test set_weights
         save_network (neural_net, *tr);
      }

      return get_trained_neuralnets();
   }

   template<class _In, class _Out, template<class _NNIn, class _NNOut> class _NN,
      template<class, class> class _Sample,
      template<class, class, template<class, class> class _TrainerSample> class _TrainAlgo>
   std::shared_ptr<flexnnet::TrainingRecord> NNetTrainer<
      _In,
      _Out,
      _NN,
      _Sample,
      _TrainAlgo>::train(NeuralNet<_In,_Out>& _nnet, const DataSet<_In, _Out, _Sample>& _trnset, const DataSet<_In, _Out, _Sample>& _vldset, const DataSet<_In, _Out, _Sample>& _tstset)
   {
      std::cout << "TrainAlgo::train_run() - entry\n";

      // Iterate through training epochs
      unsigned int max_epochs =_TrainAlgo<_In,_Out,_Sample>::get_max_epochs();
      unsigned int epoch = 0;
      for (epoch = 0; epoch < max_epochs; epoch++)
      {
         // TODO - train_run epoch
         _TrainAlgo<_In,_Out,_Sample>::train_epoch(_nnet, _trnset, _vldset, _tstset);
      }

      // TODO - implement
      return std::shared_ptr<flexnnet::TrainingRecord>(new flexnnet::TrainingRecord());
   }
}

#endif //FLEX_NEURALNET_BASICTRAINER_H_
