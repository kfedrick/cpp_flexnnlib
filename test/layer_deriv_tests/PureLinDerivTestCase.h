//
// Created by kfedrick on 5/23/19.
//

#ifndef _PURELINDERIVTESTCASE_H_
#define _PURELINDERIVTESTCASE_H_

#include "LayerDerivTestCase.h"

namespace flexnnet
{
   class PureLinDerivTestCase : public LayerDerivTestCase
   {
   public:
      void read (const std::string &_filepath);

   public:
      double gain;

   };
}

#endif //_PURELINDERIVTESTCASE_H_
