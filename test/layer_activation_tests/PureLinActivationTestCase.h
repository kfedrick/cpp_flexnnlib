//
// Created by kfedrick on 5/19/19.
//

#ifndef FLEX_NEURALNET_PURELINACTIVATIONTESTCASE_H_
#define FLEX_NEURALNET_PURELINACTIVATIONTESTCASE_H_

#include "LayerActivationTestCase.h"

namespace flexnnet
{

   class PureLinActivationTestCase : public LayerActivationTestCase
   {
   public:
      void read (const std::string &_filepath);

   public:
      double gain;

   };
}


#endif //FLEX_NEURALNET_PURELINACTIVATIONTESTCASE_H_
