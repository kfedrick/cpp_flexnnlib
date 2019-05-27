/*
 * AdaptiveCriticNet3.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ADAPTIVECRITIC_NET3_H_
#define FLEX_NEURALNET_ADAPTIVECRITIC_NET3_H_

#include "TDCNeuralNet.h"

using namespace std;

namespace flexnnet
{

   class AdaptiveCriticNet3 : public TDCNeuralNet
   {
   public:
      AdaptiveCriticNet3 ();
      virtual ~AdaptiveCriticNet3 ();

      virtual double get_reinforcement (const Pattern &_stateVec,
                                        const Pattern &_actionVec, unsigned int recurStep = 1);

   private:
      ConnectionMap conn_map;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_ADAPTIVECRITIC_NET3_H_ */
