//
// Created by kfedrick on 3/16/21.
//

#ifndef _MOCKNN_H_
#define _MOCKNN_H_

#include <FeatureVector.h>
#include <NeuralNet.h>

template<class _InType, class _OutType>
class MockNN : public flexnnet::NeuralNet<_InType, _OutType>
{
public:
   MockNN(const flexnnet::BaseNeuralNet& _nnet);
   const flexnnet::FeatureVector&
   activate(const flexnnet::FeatureVector& _nninput);

private:
   flexnnet::FeatureVector cached_input;
};

template<class _InType, class _OutType>
MockNN<_InType, _OutType>::MockNN(const flexnnet::BaseNeuralNet& _nnet) : flexnnet::NeuralNet<_InType, _OutType>(_nnet)
{

}

template<class _InType, class _OutType>
const flexnnet::FeatureVector&
MockNN<_InType, _OutType>::activate(const flexnnet::FeatureVector& _nninput)
{
   cached_input = _nninput;
   return cached_input;
}

#endif //_MOCKNN_H_
