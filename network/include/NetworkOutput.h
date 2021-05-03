//
// Created by kfedrick on 4/25/21.
//

#ifndef FLEX_NEURALNET_NETWORKOUTPUT_H_
#define FLEX_NEURALNET_NETWORKOUTPUT_H_

#include <flexnnet.h>

namespace flexnnet
{
   class NetworkOutput
   {
   public:
      virtual void set_from_nnet_encoding(const ValarrMap& _vmap) = 0;
   };
}
#endif //FLEX_NEURALNET_NETWORKOUTPUT_H_
