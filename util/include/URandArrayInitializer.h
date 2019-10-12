/*
 * URandArrayInitializer.h
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_URANDARRAYINIT_H_
#define FLEX_NEURALNET_URANDARRAYINIT_H_

#include "ArrayInitializer.h"
#include <stdlib.h>
#include <time.h>

using namespace std;

namespace flexnnet
{

   class URandArrayInitializer : public ArrayInitializer
   {
   public:
      URandArrayInitializer(double lower = -1.0, double upper = 1.0);
      virtual ~URandArrayInitializer();

      void operator()(Array<double>& arr) const;

      URandArrayInitializer* clone() const;

   private:
      double urand() const;
      double urand(double a, double b) const;

      double lower_bound;
      double upper_bound;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_URANDARRAYINIT_H_ */
