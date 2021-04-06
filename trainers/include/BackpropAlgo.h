//
// Created by kfedrick on 10/8/19.
//

#ifndef FLEX_NEURALNET_BACKPROPALGO_H_
#define FLEX_NEURALNET_BACKPROPALGO_H_

#include <cstddef>
#include <network/include/NeuralNet.h>
#include <iostream>
#include <util/include/NetworkError.h>

namespace flexnnet
{
   template<class _NNIn, class _NNOut, template<class, class> class _TData, template<class> class _ErrFunc = RMSError>
   class BackpropAlgo : public _ErrFunc<_NNOut>
   {
   protected:
      using NN_Typ_ = NeuralNet<_NNIn, _NNOut>;
      using Exemplar_Typ_ = std::tuple<_NNIn, _NNOut>;

   public:
      /**
       * Present one sample to the network from the training set and calculate
       * the appropriate weight adjustment.
       *
       * @param _nnet
       * @param _in
       * @param _out
       */
      void present_datum(size_t _epoch, NN_Typ_& _nnet, const Exemplar_Typ_& _sample);

      void update_weights(NN_Typ_& _nnet);

   private:
      void calc_weight_updates(const std::valarray<double>&, NN_Typ_& _nnet);
   };

   template<class _NNIn, class _NNOut, template<class, class> class _TData, template<class> class _ErrFunc>
   void
   BackpropAlgo<_NNIn,
                _NNOut,
                _TData,
                _ErrFunc>::present_datum(size_t _epoch, NN_Typ_& _nnet, const Exemplar_Typ_& _exemplar)
   {
      std::cout << "         Enter - BackpropAlgo::present_datum()\n";

      const _NNOut& nn_out = _nnet.activate(_exemplar.input());

      NetworkError ne = _ErrFunc<_NNOut>::error(nn_out, _exemplar.target());

      calc_weight_updates(ne.dEdy, _nnet);
   }

   template<class _NNIn, class _NNOut, template<class, class> class _TData, template<class> class _ErrFunc>
   void BackpropAlgo<_NNIn, _NNOut, _TData, _ErrFunc>::calc_weight_updates(const std::valarray<double>&, NN_Typ_& _nnet)
   {

   }

   template<class _NNIn, class _NNOut, template<class, class> class _TData, template<class> class _ErrFunc>
   void BackpropAlgo<_NNIn, _NNOut, _TData, _ErrFunc>::update_weights(NN_Typ_& _nnet)
   {

   }

}
#endif //FLEX_NEURALNET_BACKPROPALGO_H_
