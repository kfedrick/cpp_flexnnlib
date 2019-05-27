/*
 * ConstantNetworkLearningRate.h
 *
 *  Created on: May 3, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_CONSTANT_LRATE_POLICY_H_
#define FLEX_NEURALNET_CONSTANT_LRATE_POLICY_H_

#include "Array.h"
#include "LearningRatePolicy.h"

#include <iostream>
#include <vector>

using namespace std;

namespace flexnnet
{

/*
 * Learning rate policy for a layer. Responsible for providing the
 * learning rates for the biases and layer weights for a layer, and
 * for updating the learning rates according to policy. The base class
 * implements a fixed,non-adaptive learning rate policy.
 */

   class ConstantLearningRate : public LearningRatePolicy
   {
   public:
      ConstantLearningRate ();
      ConstantLearningRate (const BaseNeuralNet &_nn);
      ConstantLearningRate (const ConstantLearningRate &_nnLRPolicy);
      ~ConstantLearningRate ();

      ConstantLearningRate &operator= (
         const ConstantLearningRate &_nnLRPolicy);
   };

   inline ConstantLearningRate::ConstantLearningRate () :
      LearningRatePolicy ()
   {
      neural_net = NULL;
   }

   inline ConstantLearningRate::ConstantLearningRate (
      const BaseNeuralNet &_nn) :
      LearningRatePolicy (_nn)
   {
   }

   inline ConstantLearningRate::ConstantLearningRate (
      const ConstantLearningRate &_nnLRPolicy) :
      LearningRatePolicy (_nnLRPolicy)
   {
   }

   inline ConstantLearningRate::~ConstantLearningRate ()
   {
      // Do nothing
   }

   inline ConstantLearningRate &ConstantLearningRate::operator= (
      const ConstantLearningRate &_nnLRPolicy)
   {
      copy (_nnLRPolicy);
      return *this;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_CONSTANT_LRATE_POLICY_H_ */
