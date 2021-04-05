//
// Created by kfedrick on 6/10/19.
//

#ifndef FLEX_NEURALNET_NETSUMLAYER_H_
#define FLEX_NEURALNET_NETSUMLAYER_H_

#include "BasicLayer.h"

namespace flexnnet
{
   class NetSumLayer : public BasicLayer
   {
   protected:
      /* ********************************************************************
       * Constructors, destructors
       */
      NetSumLayer(size_t sz, const std::string& _name);
      NetSumLayer(const NetSumLayer& _netsum);

   public:
      virtual ~NetSumLayer();

   protected:

      /**
       * Calculate the net input value based on the raw input vector and weights specified in the
       * argument list and writes it into the _netin argument.
       */
      const std::valarray<double>& calc_netin(const std::valarray<double>& _rawin);

      /**
       * Calculate the derivative of the net input with respect to the weights based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      const Array2D<double>& calc_dnet_dw(const std::valarray<double>& _rawin);

      /**
       * Calculate the derivative of the net input with respect to the raw input based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      const Array2D<double>& calc_dnet_dx(const std::valarray<double>& _rawin);
   };
}

#endif //FLEX_NEURALNET_NETSUMLAYER_H_
