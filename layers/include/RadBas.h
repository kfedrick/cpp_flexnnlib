/*
 * RadBas.h
 *
 *  Created on: Mar 31, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_RADBAS_H_
#define FLEX_NEURALNET_RADBAS_H_

#include "TransferFunctor.h"
#include "NetDist.h"

namespace flexnnet
{

   class RadBas : public TransferFunctor, public NetDist
   {
   public:

      /*
       * Return transfer function type for default object name creation.
       */
      static string type()
      {
         return "RadBas";
      }

      RadBas ();

      void operator() (vector<double> &transVec, Array<double> &dAdN, vector<double> &d2AdN,
                       Array<double> &dAdB, const vector<double> &netInVec, const vector<double> &biases) const;

      RadBas *clone () const;

   private:
      mutable double spread;
      mutable double sqr_dist;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_RADBAS_H_ */
