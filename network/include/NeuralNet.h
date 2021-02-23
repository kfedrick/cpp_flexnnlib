//
// Created by kfedrick on 9/17/19.
//

#ifndef FLEX_NEURALNET_NEURALNET_H_
#define FLEX_NEURALNET_NEURALNET_H_

#include "BasicNeuralNet.h"
#include "BasicNeuralNetSerializer.h"

namespace flexnnet
{
   template<class _InType, class _OutType>
   class NeuralNet : public BasicNeuralNet, public BasicNeuralNetSerializer
   {

   public:
      NeuralNet(const std::vector<std::shared_ptr<OldNetworkLayer>>& layers, bool _recurrent, const std::string& _name = "BasicNeuralNet");
      virtual ~NeuralNet();

   public:

      const _OutType& activate(const _InType& _indatum);
      const _InType& get_network_input(void) const;

   private:
      // network_output_pattern - Cached value for the most recent network activation
      //
      _OutType network_output_pattern;

      // network_input - Cached value for the most recent network input value
      //
      _InType network_input;
   };

   template<class _InType, class _OutType>
   const _OutType& NeuralNet<_InType, _OutType>::activate(const _InType& _xdatum)
   {
      /*
       * Activate all network layers
       *
      for (int i = 0; i < network_layers.size (); i++)
      {
         // Get a network basiclayer
         OldNetworkLayer& basiclayer = *network_layers[i];

         const std::valarray<double> &invec = basiclayer.coelesce_input(_xdatum);
         basiclayer.activate (invec);
      }

      // Next add basiclayer outputs
      for (auto nlayer : network_layers)
      {
         if (nlayer->is_output_layer ())
         {
            const std::valarray<double> &layer_outputv = (*nlayer)();
            network_output_pattern.set_weights (nlayer->name (), layer_outputv);
         }
      }
       */
      return network_output_pattern;
   }

   template<class _InType, class _OutType>
   NeuralNet<_InType,
             _OutType>::NeuralNet(const std::vector<std::shared_ptr<OldNetworkLayer>>& _layers, bool _recurrent, const std::string& _name)
      : BasicNeuralNet(_layers, _recurrent, _name)
   {

   }

   template<class _InType, class _OutType>
   NeuralNet<_InType, _OutType>::~NeuralNet()
   {

   }

   /*
   template<class _InType, class _OutType>
   const _OutType & NeuralNet<_InType, _OutType>::activate (const _InType &_indatum)
   {
      return BasicNeuralNet::activate(_indatum);
   }
    */

   template<class _InType, class _OutType>
   const _InType& NeuralNet<_InType, _OutType>::get_network_input(void) const
   {
      return network_input;
   }
}

#endif //FLEX_NEURALNET_NEURALNET_H_
