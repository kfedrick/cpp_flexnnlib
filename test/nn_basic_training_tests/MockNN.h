//
// Created by kfedrick on 3/16/21.
//

#ifndef _MOCKNN_H_
#define _MOCKNN_H_

#include <NeuralNet.h>
#include <FeatureSetImpl.h>
#include <RawFeature.h>

template<class _InType, class _OutType=flexnnet::FeatureSetImpl<std::tuple<flexnnet::RawFeature<1>>>>
class MockNN : public flexnnet::NeuralNet<_InType, _OutType>
{
public:
   MockNN(const flexnnet::BaseNeuralNet& _nnet);
   const flexnnet::NNFeatureSet<_OutType>&
   activate(const _InType& _nninput);

private:
   flexnnet::FeatureSetImpl<std::tuple<flexnnet::RawFeature<1>>> cached_input;
};

template<class _InType, class _OutType>
MockNN<_InType, _OutType>::MockNN(const flexnnet::BaseNeuralNet& _nnet) : flexnnet::NeuralNet<_InType, _OutType>(_nnet)
{
}

template<class _InType, class _OutType>
const flexnnet::NNFeatureSet<_OutType>&
MockNN<_InType, _OutType>::activate(const _InType& _nninput)
{
   std::cout << "activate ordered layer count = 0 for Mock" << "\n";

   //cached_input = _nninput;
   //return cached_input;
   return this->value();
}

#endif //_MOCKNN_H_
