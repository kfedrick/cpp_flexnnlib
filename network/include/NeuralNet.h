//
// Created by kfedrick on 9/17/19.
//

#ifndef FLEX_NEURALNET_NEURALNET_H_
#define FLEX_NEURALNET_NEURALNET_H_

#include "BaseNeuralNet.h"

namespace flexnnet
{
   template<class InTyp, class OutTyp>
   class NeuralNet : public BaseNeuralNet
   {

   public:
      NeuralNet(const BaseNeuralNet& _nnet);
      virtual ~NeuralNet();

   public:

      const OutTyp&
      activate(const InTyp& _nninput);

      const InTyp&
      get_network_input(void) const;

      const OutTyp&
      value(void) const;

   private:
      // network_output - Cached value for the most recent network activation
      OutTyp network_output;

      // network_input - Cached value for the most recent network input value
      InTyp network_input;
   };

   template<class InTyp, class OutTyp>
   const OutTyp&
   NeuralNet<InTyp, OutTyp>::activate(const InTyp& _nninput)
   {
      BaseNeuralNet::activate(_nninput.value_map());

      network_output.parse(value_map());
      return network_output;
   }

   template<class InTyp, class OutTyp>
   NeuralNet<InTyp,
             OutTyp>::NeuralNet(const BaseNeuralNet& _nnet)
      : BaseNeuralNet(_nnet)
   {

   }

   template<class InTyp, class OutTyp>
   NeuralNet<InTyp, OutTyp>::~NeuralNet()
   {

   }

   template<class InTyp, class OutTyp>
   const InTyp&

   NeuralNet<InTyp, OutTyp>::get_network_input(void) const
   {
      return network_input;
   }

   template<class InTyp, class OutTyp>
   const OutTyp&
   NeuralNet<InTyp, OutTyp>::value(void) const
   {
      return network_output;
   }
}

#endif //FLEX_NEURALNET_NEURALNET_H_
