//
// Created by kfedrick on 9/17/19.
//

#ifndef FLEX_NEURALNET_NEURALNET_H_
#define FLEX_NEURALNET_NEURALNET_H_

#include <BaseNeuralNet.h>
#include <NetworkOutput.h>
#include <NNFeatureSet.h>
#include <ValueMapFeatureSet.h>

namespace flexnnet
{
   template<class InTyp, class OutTyp=NetworkOutput>
   class NeuralNet : public BaseNeuralNet
   {
   public:
      NeuralNet(const BaseNeuralNet& _nnet);
      NeuralNet(const NeuralNet<InTyp,OutTyp>& _nnet);
      virtual ~NeuralNet();

      NeuralNet<InTyp,OutTyp>& operator=(const NeuralNet<InTyp,OutTyp>& _nnet);

   protected:
      void copy(const NeuralNet<InTyp,OutTyp>& _nnet);

   public:

      const flexnnet::NNFeatureSet<OutTyp>&
      activate(const InTyp& _nninput);

      const InTyp&
      get_network_input(void) const;

      const flexnnet::NNFeatureSet<OutTyp>&
      value(void) const;

   private:
      // network_output - Cached vectorize for the most recent network activation
      //OutTyp network_output;

      flexnnet::NNFeatureSet<OutTyp> network_output;

      // network_input - Cached vectorize for the most recent network input vectorize
      InTyp network_input;
   };

   template<class InTyp, class OutTyp>
   NeuralNet<InTyp,
             OutTyp>::NeuralNet(const BaseNeuralNet& _nnet)
      : BaseNeuralNet(_nnet)
   {
      //std::cout << "NeuralNet::NeuralNet(const BaseNeuralNet&)\n";

      //network_output = OutTyp(network_output_layers);

      //std::cout << "NN<>:contructor layer id " << network_output_layers[0]->name() << "\n";
      network_output = flexnnet::NNFeatureSet<OutTyp>(network_output_layers);
      //std::cout << "NN<>.constructor nnout.name() " << network_output.value_map().begin()->first << "\n";
   }

   template<class InTyp, class OutTyp>
   NeuralNet<InTyp, OutTyp>::NeuralNet(const NeuralNet<InTyp,OutTyp>& _nnet) : BaseNeuralNet(_nnet)
   {
      //std::cout << "NeuralNet::NeuralNet(const NeuralNet&)\n";
      copy(_nnet);

      network_output = flexnnet::NNFeatureSet<OutTyp>(network_output_layers);
      //std::cout << "NN<>.copy constructor nnout.name() " << _nnet.network_output.value_map().begin()->first << "\n";

   }

   template<class InTyp, class OutTyp>
   NeuralNet<InTyp,OutTyp>& NeuralNet<InTyp, OutTyp>::operator=(const NeuralNet<InTyp,OutTyp>& _nnet)
   {
      std::cout << "NN<>::operator= - ENTRY\n";

      //copy(_nnet);
      network_output = flexnnet::NNFeatureSet<OutTyp>(network_output_layers);
      return *this;
   }

   template<class InTyp, class OutTyp>
   void NeuralNet<InTyp, OutTyp>::copy(const NeuralNet<InTyp,OutTyp>& _nnet)
   {
      //std::cout << "NN<>.copy nnout.name() " << _nnet.network_output.value_map().begin()->first << "\n";

      network_output = _nnet.network_output;
      //network_input = _nnet.network_input;
   }

   template<class InTyp, class OutTyp>
   NeuralNet<InTyp, OutTyp>::~NeuralNet()
   {
   }

   template<class InTyp, class OutTyp>
   const flexnnet::NNFeatureSet<OutTyp>&
   NeuralNet<InTyp, OutTyp>::activate(const InTyp& _nninput)
   {
      const ValarrMap& vm = BaseNeuralNet::activate(_nninput.value_map());

/*      std::cout << "nn inputs : ";
      for (auto entry : _nninput.value_map())
         std::cout << entry.first << " ";
      std::cout << "\n";*/
      //std::cout << "NN<>.activate nnout.name() " << vm.begin()->first << " " << vm.begin()->second[0] << "\n";


      network_output.activate();

      //std::cout << "NN<>.activate nnout.name() " << network_output.value_map().begin()->first << "\n";

      return network_output;
   }

   template<class InTyp, class OutTyp>
   const InTyp&

   NeuralNet<InTyp, OutTyp>::get_network_input(void) const
   {
      return network_input;
   }

   template<class InTyp, class OutTyp>
   const flexnnet::NNFeatureSet<OutTyp>&
   NeuralNet<InTyp, OutTyp>::value(void) const
   {
      return network_output;
   }
}

#endif //FLEX_NEURALNET_NEURALNET_H_
