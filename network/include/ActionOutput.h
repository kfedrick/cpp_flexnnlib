//
// Created by kfedrick on 5/21/21.
//

#ifndef FLEX_NEURALNET_ACTIONOUTPUT_H_
#define FLEX_NEURALNET_ACTIONOUTPUT_H_

#include "NetworkOutput.h"

namespace flexnnet
{
   template<typename Action>
   class ActionOutput : public NetworkOutput
   {
      virtual const Action& get_action() const = 0;
   };
}

#endif //FLEX_NEURALNET_ACTIONOUTPUT_H_
