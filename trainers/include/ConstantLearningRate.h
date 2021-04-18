/*
 * ConstantNetworkLearningRate.h
 *
 *  Created on: May 3, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_CONSTANT_LRATE_POLICY_H_
#define FLEX_NEURALNET_CONSTANT_LRATE_POLICY_H_

#include "Array2D.h"
#include "LearningRatePolicy.h"

#include <iostream>
#include <vector>

namespace flexnnet
{

/*
 * Learning rate policy for a layer. Responsible for providing the
 * learning rates for the biases and layer weights for a layer, and
 * for updating the learning rates according to policy. The base class
 * implements a fixed,non-adaptive learning rate policy.
 */

class ConstantLearningRate: public LearningRatePolicy
{
public:
   ConstantLearningRate();
   ConstantLearningRate(BaseNeuralNet& _nnet);
   ConstantLearningRate(const ConstantLearningRate& _nnLRPolicy);
   ~ConstantLearningRate();

   ConstantLearningRate& operator=(
         const ConstantLearningRate& _nnLRPolicy);

private:
   double DEFAULT_LEARNING_RATE{0.001};
};

inline ConstantLearningRate::ConstantLearningRate() :
   LearningRatePolicy()
{
}

inline ConstantLearningRate::ConstantLearningRate(
   BaseNeuralNet& _nnet) :
   LearningRatePolicy(_nnet)
{
   set_learning_rate(DEFAULT_LEARNING_RATE);
}

inline ConstantLearningRate::ConstantLearningRate(
      const ConstantLearningRate& _nnLRPolicy) :
   LearningRatePolicy(_nnLRPolicy)
{
}

inline ConstantLearningRate::~ConstantLearningRate()
{
   // Do nothing
}

inline ConstantLearningRate& ConstantLearningRate::operator=(
      const ConstantLearningRate& _nnLRPolicy)
{
   copy(_nnLRPolicy);
   return *this;
}

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_CONSTANT_LRATE_POLICY_H_ */
