//
// Created by kfedrick on 3/16/21.
//

#ifndef _MOCKNN_H_
#define _MOCKNN_H_

#include <NeuralNet.h>

template<class _InType, class _OutType>
class MockNN : public flexnnet::NeuralNet<_InType, _OutType>
{
public:
   MockNN(const flexnnet::BaseNeuralNet& _nnet);
   const _OutType&
   activate(const _InType& _nninput);

private:
   flexnnet::NNFeatureSet<_OutType> network_output;
   _InType cached_input;
};

template<class _InType, class _OutType>
MockNN<_InType, _OutType>::MockNN(const flexnnet::BaseNeuralNet& _nnet) : flexnnet::NeuralNet<_InType, _OutType>(_nnet)
{

}

template<class _InType, class _OutType>
const _OutType&
MockNN<_InType, _OutType>::activate(const _InType& _nninput)
{
   cached_input = _nninput;
   network_output = _nninput;
   return network_output;
}

#endif //_MOCKNN_H_
