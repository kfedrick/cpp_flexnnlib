//
// Created by kfedrick on 5/8/21.
//

#ifndef FLEX_NEURALNET_ACTIONVIEW_H_
#define FLEX_NEURALNET_ACTIONVIEW_H_

#include <flexnnet.h>

namespace flexnnet
{
   template<typename ActionE>
   class ActionView
   {
   public:
      /**
       * Return the action indicated by the current action state.
       *
       * @return
       */
      virtual ActionE decode_action(void) = 0;
   };
}

#endif //FLEX_NEURALNET_ACTIONVIEW_H_
