//
// Created by kfedrick on 4/25/21.
//

#ifndef FLEX_NEURALNET_NETWORKINPUT_H_
#define FLEX_NEURALNET_NETWORKINPUT_H_

#include <flexnnet.h>

namespace flexnnet
{
   class NetworkInput
   {
   public:
      virtual const ValarrMap& get_nnet_encoding() = 0;
   };
}
#endif //FLEX_NEURALNET_NETWORKINPUT_H_
