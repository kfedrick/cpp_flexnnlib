//
// Created by kfedrick on 6/10/19.
//

#ifndef FLEX_NEURALNET_EUCLIDEANDISTLAYER_H_
#define FLEX_NEURALNET_EUCLIDEANDISTLAYER_H_

#include "BasicLayer.h"

namespace flexnnet
{
   class EuclideanDistLayer : public BasicLayer
   {
   protected:
      /* ********************************************************************
       * Constructors, destructors
       */
      EuclideanDistLayer(size_t _sz, const std::string& _name);

   public:
      virtual ~EuclideanDistLayer();

   protected:

      /**
       * Calculate the net input value based on the raw input vector and weights specified in the
       * argument list and writes it into the _netin argument.
       */
      void calc_netin(const std::valarray<double>& _rawinv, std::valarray<double>& _netin);

      /**
       * Calculate the derivative of the net input with respect to the weights based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      void calc_dnet_dw(const LayerState& _lstate, Array2D<double>& _dnetdw);

      /**
       * Calculate the derivative of the net input with respect to the raw input based on the raw
       * input vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
      void calc_dnet_dx(const LayerState& _lstate, Array2D<double>& _dnetdx);

   private:
      std::valarray<double> squared_euclidean_dist;
      std::valarray<double> spread;
   };
}

#endif //FLEX_NEURALNET_EUCLIDEANDISTLAYER_H_
