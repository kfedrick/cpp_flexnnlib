//
// Created by kfedrick on 5/15/19.
//

#ifndef _LAYERINFO_H_
#define _LAYERINFO_H_

#include "Array.h"

namespace flexnnet
{
   class LayerInfo
   {
   public:
      std::string name;
      unsigned int output_size;
      unsigned int input_size;
      Array<double> weights;
   };
}

#endif //_LAYERINFO_H_
