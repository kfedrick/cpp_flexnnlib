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

   const std::valarray<double>&
   backprop(const flexnnet::FeatureVector& _errormap);

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
   std::cout << "activate ordered layer count = 0 for Mock" << "\n";

   cached_input = _nninput;
   return cached_input;
}

template<class _InType, class _OutType>
const std::valarray<double>&
MockNN<_InType, _OutType>::backprop(const flexnnet::FeatureVector& _errorv)
{
   static std::valarray<double> e;
   return e;
}

#endif //_MOCKNN_H_
