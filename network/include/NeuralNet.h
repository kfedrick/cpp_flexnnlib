//
// Created by kfedrick on 9/17/19.
//

#ifndef FLEX_NEURALNET_NEURALNET_H_
#define FLEX_NEURALNET_NEURALNET_H_

#include "BaseNeuralNet.h"

namespace flexnnet
{
   template<class _InType, class _OutType>
   class NeuralNet : public BaseNeuralNet
   {

   public:
      NeuralNet(const BaseNeuralNet& _nnet);
      virtual ~NeuralNet();

   public:

      const _OutType&
      activate(const _InType& _nninput);


      const _InType&
      get_network_input(void) const;

   private:
      // network_output_pattern - Cached value for the most recent network activation
      //
      _OutType network_output_pattern;

      // network_input - Cached value for the most recent network input value
      //
      _InType network_input;
   };

   template<class _InType, class _OutType>
   const _OutType&
   NeuralNet<_InType, _OutType>::activate(const _InType& _nninput)
   {
      BaseNeuralNet::activate(_nninput.value_map());

      network_output_pattern.parse(value_map());
      return network_output_pattern;
   }

   template<class _InType, class _OutType>
   NeuralNet<_InType,
             _OutType>::NeuralNet(const BaseNeuralNet& _nnet)
      : BaseNeuralNet(_nnet)
   {

   }

   template<class _InType, class _OutType>
   NeuralNet<_InType, _OutType>::~NeuralNet()
   {

   }

   template<class _InType, class _OutType>
   const _InType&
   NeuralNet<_InType, _OutType>::get_network_input(void) const
   {
      return network_input;
   }
}

#endif //FLEX_NEURALNET_NEURALNET_H_
