/*
 * PatternSequence.h
 *
 *  Created on: Mar 28, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_PATTERNSEQUENCE_H_
#define FLEX_NEURALNET_PATTERNSEQUENCE_H_

#include <vector>
#include "deprecated/Pattern.h"

using namespace std;

namespace flexnnet
{

   class PatternSequence : public vector<Pattern>
   {
   public:
      PatternSequence ();
      PatternSequence (int sz);
      PatternSequence (unsigned int sz);
      PatternSequence (const vector<Pattern> &vec);
   };

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_PATTERNSEQUENCE_H_ */
