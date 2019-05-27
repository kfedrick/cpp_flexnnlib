/*
 * OutputErrorFunctor.h
 *
 *  Created on: Feb 5, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_OUTPUTERRORFUNCTOR_H_
#define FLEX_NEURALNET_OUTPUTERRORFUNCTOR_H_

#include <vector>

using namespace std;
namespace flexnnet
{

   class OutputErrorFunctor
   {
   public:
      OutputErrorFunctor ();
      virtual ~OutputErrorFunctor ();

      virtual void
      operator() (double &error, vector<double> &gradient, const vector<double> &outVec, const vector<double> &targetVec);
      virtual OutputErrorFunctor *clone () const;
   };

}

#endif /* FLEX_NEURALNET_OUTPUTERRORFUNCTOR_H_ */
