//
// Created by kfedrick on 5/11/19.
//

#ifndef FLEX_NEURALNET_NETSUMLAYER_H_
#define FLEX_NEURALNET_NETSUMLAYER_H_

#include "Array.h"

namespace flexnnet
{
   class MyNetSum
   {
   public:
      MyNetSum(unsigned int _netvec_sz);
      ~MyNetSum();

   public:
      /**
       * Calculate the net input value based on the raw input vector and weights specified in the
       * argument list and writes it into the _netin argument.
       */
       const std::vector<double>& calc_netin(const std::vector<double> &_rawin, const Array<double>& _weights);

      /**
       * Calculate the derivative of the net input with respect to the weights based on the raw
       * input std::vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
       const Array<double>& calc_dNdW(const std::vector<double> &_netin, const std::vector<double> &_rawin, const Array<double>& _weights);

      /**
       * Calculate the derivative of the net input with respect to the raw input based on the raw
       * input std::vector and weights specified in the argument list and writes it into the _dNdW argument.
       */
       const Array<double>& calc_dNdI(const std::vector<double> &_netin, const std::vector<double> &_rawin, const Array<double>& _weights);

   protected:
      void resize(unsigned int _layer_sz, unsigned int _rawin_sz);

   private:
      const unsigned int netinvec_size;
      unsigned int rawinvec_size;

   private:
      // Cached values of net input from most recent activation.
      std::vector<double> netin;

      // Cached value of derivative of net input wrt weights from most recent activation
      Array<double> dNdW;

      // Cached value of derivative of net input wrt external input from most recent activation
      Array<double> dNdI;
   };

   inline void MyNetSum::resize(unsigned int _layer_sz, unsigned int _rawin_sz)
   {
      rawinvec_size = _rawin_sz;

      dNdW.resize(netinvec_size, rawinvec_size+1);
      dNdI.resize(netinvec_size, rawinvec_size+1);
   }

}

#endif //FLEX_NEURALNET_NETSUMLAYER_H_
