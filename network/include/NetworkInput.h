//
// Created by kfedrick on 4/25/21.
//

#ifndef FLEX_NEURALNET_NETWORKINPUT_H_
#define FLEX_NEURALNET_NETWORKINPUT_H_

#include <flexnnet.h>
#include <Vectorizable.h>

namespace flexnnet
{
   class NetworkInput
   {
   public:
      virtual const ValarrMap& value_map() const = 0;
   };
}
#endif //FLEX_NEURALNET_NETWORKINPUT_H_
