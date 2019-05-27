/*
 * PatternSequence.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: kfedrick
 */

#include "PatternSequence.h"

namespace flexnnet
{

   PatternSequence::PatternSequence () : vector<Pattern> ()
   {}
   PatternSequence::PatternSequence (int sz) : vector<Pattern> (sz)
   {}
   PatternSequence::PatternSequence (unsigned int sz) : vector<Pattern> (sz)
   {}
   PatternSequence::PatternSequence (const vector<Pattern> &vec) : vector<Pattern> (vec)
   {}

} /* namespace flexnnet */
