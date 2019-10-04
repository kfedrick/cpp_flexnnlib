//
// Created by kfedrick on 9/22/19.
//

#ifndef FLEX_NEURALNET_FATRAINERALGO_H_
#define FLEX_NEURALNET_FATRAINERALGO_H_

#include <cstddef>
#include <memory>

#include "TrainerUtils.h"
#include "TrainerConfig.h"
#include "FuncApproxEvaluator.h"

#include "BasicNeuralNet.h"
#include "ExemplarSet.h"
#include "Exemplar.h"

namespace flexnnet
{
   template<class _NNIn, class _NNOut, class _ErrFunc, template<class,class,class> class _Eval>
   class FATrainer : public TrainerUtils, public TrainerConfig, public _Eval<_NNIn, _NNOut, _ErrFunc>
   {
   protected:
      using NN_Typ_ = NeuralNet<_NNIn, _NNOut>;
      using DataSet_Typ_ = ExemplarSet<_NNIn, _NNOut>;
      using Exemplar_Typ_ = Exemplar<_NNIn, _NNOut>;

   public:

      /**
       * Train neural network for up to the maximum specified epochs or until the
       * convergence criteria is met.
       *
       * @param _nnet
       * @param _trnset
       * @param _tstset
       * @return
       */
      void train(NN_Typ_& _nnet, const DataSet_Typ_& _trnset, const DataSet_Typ_& _tstset = DataSet_Typ_());

      /**
       * Train neural network for up to the maximum specified epochs or until the
       * convergence criteria is met. Use early stopping to regularize - upon return
       * set the network weights to those with the best performance on the
       * validation set.
       *
       * @param _nnet
       * @param _trnset
       * @param _vldset
       * @param _tstset
       * @return
       */

      std::shared_ptr<flexnnet::TrainingRecord>
      train_with_validation_set(NN_Typ_& _nnet, const DataSet_Typ_& _trnset, const DataSet_Typ_& _vldset, const DataSet_Typ_& _tstset = DataSet_Typ_());

   protected:

      /**
       * Perform one training epoch by presenting each sample in the training set
       * to the network and updating the weights.
       *
       * @tparam _NNIn
       * @tparam _NNOut
       * @tparam _NN
       * @tparam _Exemplar
       * @param _nnet
       * @param _trnset
       * @param _vldset
       * @param _tstset
       */
      void train_epoch(_index_typ _epoch, NN_Typ_& _nnet, const DataSet_Typ_& _trnset, const DataSet_Typ_& _vldset, const DataSet_Typ_& _tstset = DataSet_Typ_());

      /**
       *
       * @tparam _NNIn
       * @tparam _NNOut
       * @tparam _NN
       * @tparam _Exemplar
       * @param _nnet
       * @param _sample
       */
      void train_sample(_index_typ _epoch, NN_Typ_& _nnet, const Exemplar_Typ_& _sample);

      /**
       * Present one sample to the network from the training set and calculate
       * the appropriate weight adjustment.
       *
       * @param _nnet
       * @param _in
       * @param _out
       */
      virtual void train_exemplar(BasicNeuralNet& _nnet, const Datum& _in, const Datum& _target);

      virtual void update_weights(void);
   };

   template<class _NNIn, class _NNOut, class _ErrFunc, template<class,class, class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _ErrFunc, _Eval>::train(NN_Typ_& _nnet, const DataSet_Typ_& _trnset, const DataSet_Typ_& _tstset)
   {
      std::cout << "FATrainer::train() - entry\n";

      double trn_perf, vld_perf, tst_perf;

      ExemplarSet<_NNIn, _NNOut> dummy_vldset = DataSet_Typ_();

      // Iterate through training epochs
      _index_typ max_epochs = get_max_epochs();
      _index_typ epoch = 0;
      for (epoch = 0; epoch < max_epochs; epoch++)
      {
         train_epoch(epoch, _nnet, _trnset, dummy_vldset, _tstset);

         //trn_perf = myeval.evaluate_performance(_nnet, _trnset);

         // TODO - check whether to fail back if adjustable learning rate
      }

      //return std::shared_ptr<flexnnet::TrainingRecord>(new flexnnet::TrainingRecord());
   }

   template<class _NNIn, class _NNOut, class _ErrFunc, template<class,class,class> class _Eval>
   std::shared_ptr<flexnnet::TrainingRecord>
   FATrainer<_NNIn, _NNOut, _ErrFunc, _Eval>::train_with_validation_set(NN_Typ_& _nnet, const DataSet_Typ_& _trnset, const DataSet_Typ_& _vldset, const DataSet_Typ_& _tstset)
   {
      std::cout << "Enter - FATrainer::train()\n";

      double trn_perf, vld_perf, tst_perf;

      // Iterate through training epochs
      _index_typ max_epochs = get_max_epochs();
      _index_typ epoch = 0;
      for (epoch = 0; epoch < max_epochs; epoch++)
      {
         train_epoch(_nnet, _trnset, epoch);

         //trn_perf = calculate_performance(_trnset);
         //vld_perf = calculate_performance(_vldset);

         // TODO - check whether to fail back if adjustable learning rate
      }

      return std::shared_ptr<flexnnet::TrainingRecord>(new flexnnet::TrainingRecord());
   }

   template<class _NNIn, class _NNOut, class _ErrFunc, template<class,class,class> class _Eval>
   void
   FATrainer<_NNIn, _NNOut, _ErrFunc, _Eval>::train_epoch(_index_typ _epoch, NN_Typ_& _nnet, const DataSet_Typ_& _trnset, const DataSet_Typ_& _vldset, const DataSet_Typ_& _tstset)
   {
      std::cout << "Enter - FATrainer::train_epoch()\n";

      // Iterate through all samples in the training set
      _index_typ sample_ndx = 0;
      for (auto asample : _trnset)
      {
         std::cout << "FATrainer::train() - sample " << sample_ndx << "\n";
         train_sample(sample_ndx, _nnet, asample);

         // If training in online or mini-batch mode, update now.
         if (batch_size() > 0 && sample_ndx % batch_size() == 0);
         update_weights();

         sample_ndx++;
      }

      // If training in batch mode, update weights now
      if (batch_size() == 0)
         update_weights();
   }

   template<class _NNIn, class _NNOut, class _ErrFunc, template<class,class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _ErrFunc, _Eval>::train_sample(_index_typ _epoch, NN_Typ_& _nnet, const Exemplar_Typ_& _sample)
   {
      std::cout << "Enter - FATrainer::train_sample()\n";
   }

   template<class _NNIn, class _NNOut, class _ErrFunc, template<class,class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _ErrFunc, _Eval>::train_exemplar(BasicNeuralNet& _nnet, const Datum& _in, const Datum& _out)
   {
   }

   template<class _NNIn, class _NNOut, class _ErrFunc, template<class,class,class> class _Eval>
   void FATrainer<_NNIn, _NNOut, _ErrFunc, _Eval>::update_weights(void)
   {
      std::cout << "Enter - FATrainer::update_weights()\n";
   }

}

#endif //_FATRAINERALGO_H_
