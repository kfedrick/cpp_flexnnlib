/*
 * ArrayInitializer.h
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ARRAYINITI_H_
#define FLEX_NEURALNET_ARRAYINITI_H_

#include <Array.h>

namespace flexnnet
{

   class ArrayInitializer
   {
   public:

      virtual ~ArrayInitializer ();

      virtual void operator() (Array<double> &arr) const;

      virtual ArrayInitializer *clone () const;
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_ARRAYINITI_H_ */
