//
// Created by kfedrick on 9/22/19.
//

#ifndef FLEX_NEURALNET_FA2TRAINER_H_
#define FLEX_NEURALNET_FA2TRAINER_H_

#include "FATrainer.h"
#include "SumSquaredError.h"

namespace flexnnet
{
   class FA2Trainer : public FATrainer<Datum,Datum,SumSquaredError,FuncApproxEvaluator>
   {
   public:

   protected:
      virtual void train_exemplar (BasicNeuralNet &_nnet, const Datum &_in, const Datum &_out);
   };

   inline
   void FA2Trainer::train_exemplar (BasicNeuralNet &_nnet, const Datum &_in, const Datum &_out)
   {
      std::cout << "FA2Trainer::train_sample() - entry\n";
   }
}

#endif //FLEX_NEURALNET_FA2TRAINER_H_
