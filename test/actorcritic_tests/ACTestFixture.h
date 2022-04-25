//
// Created by kfedrick on 5/30/21.
//

#ifndef FLEX_NEURALNET_ACTESTFIXTURE_H_
#define FLEX_NEURALNET_ACTESTFIXTURE_H_

#include <gtest/gtest.h>
#include <CommonTestFixtureFunctions.h>
#include "test/include/BaseNNActivationTestFixture.h"

#include <flexnnet.h>

#include <PureLin.h>
#include <TanSig.h>
#include <LogSig.h>
#include <RadBas.h>
#include <SoftMax.h>
#include <NetworkLayer.h>
#include <NetworkLayerImpl.h>
#include <BaseNeuralNet.h>
#include <NeuralNet.h>
#include <BaseActorCriticNetwork.h>

#include <Feature.h>
#include <RawFeature.h>
#include <RawFeatureSet.h>
#include <Reinforcement.h>
#include <picojson.h>
#include <fstream>

#include "TestAction.h"
#include <ActionSet.h>
#include "SteeringActionFeature.h"

using flexnnet::Array2D;

using flexnnet::PureLin;
using flexnnet::TanSig;
using flexnnet::LogSig;
using flexnnet::RadBas;
using flexnnet::SoftMax;
using flexnnet::NetworkLayerImpl;
using flexnnet::LayerConnRecord;

using flexnnet::NeuralNetTopology;
using flexnnet::BaseNeuralNet;
using flexnnet::NeuralNet;
using flexnnet::BaseActorCriticNetwork;

using flexnnet::Feature;
using flexnnet::RawFeature;
using flexnnet::FeatureSetImpl;
using flexnnet::RawFeatureSet;
using flexnnet::Reinforcement;

template<typename T>
class ACTestFixture : public CommonTestFixtureFunctions, public BaseNNActivationTestFixture<T>, public ::testing::Test
{
public:

   virtual void
   SetUp()
   {}

   virtual void
   TearDown()
   {}

   template<typename I, typename O>
   NeuralNet<I,O> create_critic(const I& _isample);

   template<typename I, typename O>
   NeuralNet<I,O> create_actor(const I& _isample, size_t _actionsz=1);

   template<typename S, typename A, typename SA, size_t N>
   BaseActorCriticNetwork<S,A,N> create_actorcritic(const S& _is, const SA& _sa, size_t _actionsz=1);
};
TYPED_TEST_CASE_P (ACTestFixture);

class InputFeatures : public RawFeatureSet<3>
{
public:
   InputFeatures() : RawFeatureSet<3>({"I1"}) {};
};

typedef decltype(std::tuple_cat(std::declval<InputFeatures>().get_features(),
                                std::declval<TestAction>().get_features())) StateActionTuple;
typedef FeatureSetImpl<StateActionTuple> StateAction;

typedef decltype(std::tuple_cat(std::declval<RawFeatureSet<14>>().get_features(),
                                std::declval<ActionSet<SteeringActionFeature>>().get_features())) DerbyStateActionTuple0;
typedef decltype(FeatureSetImpl<DerbyStateActionTuple0>({"state", "action"})) DerbyStateAction0;

typedef decltype(std::tuple_cat(std::declval<RawFeatureSet<23>>().get_features(),
                                std::declval<ActionSet<SteeringActionFeature>>().get_features())) DerbyStateActionTuple;
typedef FeatureSetImpl<DerbyStateActionTuple> DerbyStateAction;

typedef decltype(std::tuple_cat(std::declval<RawFeatureSet<(10+2)*(10+1)>>().get_features(),
                                std::declval<ActionSet<SteeringActionFeature>>().get_features())) DerbyStateActionTuple2;
typedef FeatureSetImpl<DerbyStateActionTuple2> DerbyStateAction2;

template<typename T>
template<typename I, typename O>
NeuralNet<I,O> ACTestFixture<T>::create_critic(const I& _isample)
{
   std::shared_ptr<NetworkLayerImpl<TanSig>> ol_ptr =
      std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(1, "R", TanSig::DEFAULT_PARAMS, true));
   std::shared_ptr<NetworkLayerImpl<T>> hl_ptr =
      std::make_shared<NetworkLayerImpl<T>>(NetworkLayerImpl<T>(2, "critic-hidden", T::DEFAULT_PARAMS, false));

   auto& names = _isample.get_feature_names();
   for (int ndx=0; ndx<_isample.size();ndx++)
      hl_ptr->add_external_input_field(names[ndx], _isample.size(ndx));

   ol_ptr->add_connection("activation", hl_ptr, LayerConnRecord::Forward);
   hl_ptr->add_connection("backprop", ol_ptr, LayerConnRecord::Forward);

   NeuralNetTopology topo;
   topo.network_layers[hl_ptr->name()] = hl_ptr;
   topo.network_layers[ol_ptr->name()] = ol_ptr;

   topo.network_output_layers.push_back(ol_ptr);

   topo.ordered_layers.push_back(hl_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::NeuralNet<I, O> critic(topo);

   critic.initialize_weights();

   return critic;
}

template<typename T>
template<typename I, typename O>
NeuralNet<I,O> ACTestFixture<T>::create_actor(const I& _isample, size_t _osz)
{
   //std::cout << "create_actor() ENTRY\n";

   std::shared_ptr<NetworkLayerImpl<T>> ol_ptr =
      std::make_shared<NetworkLayerImpl<T>>(NetworkLayerImpl<T>(_osz, "action", T::DEFAULT_PARAMS, true));
   std::shared_ptr<NetworkLayerImpl<TanSig>> hl_ptr =
      std::make_shared<NetworkLayerImpl<TanSig>>(NetworkLayerImpl<TanSig>(2, "actor-hidden", TanSig::DEFAULT_PARAMS, false));

   auto& names = _isample.get_feature_names();
   for (int ndx=0; ndx<_isample.size();ndx++)
      hl_ptr->add_external_input_field(names[ndx], _isample.size(ndx));

   ol_ptr->add_connection("activation", hl_ptr, LayerConnRecord::Forward);
   hl_ptr->add_connection("backprop", ol_ptr, LayerConnRecord::Forward);

   NeuralNetTopology topo;
   topo.network_layers[hl_ptr->name()] = hl_ptr;
   topo.network_layers[ol_ptr->name()] = ol_ptr;

   topo.network_output_layers.push_back(ol_ptr);

   topo.ordered_layers.push_back(hl_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::NeuralNet<I, O> actor(topo);

   actor.initialize_weights();

   //std::cout << "create_actor() EXIT\n";

   return actor;
}

template<typename T>
template<typename S, typename A, typename SA, size_t N>
BaseActorCriticNetwork<S,A,N> ACTestFixture<T>::create_actorcritic(const S& _is, const SA& _sa, size_t _actionsz)
{
   NeuralNet<S, A> actor = create_actor<S, A>(_is, _actionsz);
   NeuralNet<SA, Reinforcement<1>> critic = create_critic<SA, Reinforcement<1>>(_sa);

   BaseActorCriticNetwork<S, A, 1> acnn;//(actor,critic);

   return acnn;
}

typedef ::testing::Types<flexnnet::TanSig> MyTypes;

#endif // FLEX_NEURALNET_ACTESTFIXTURE_H_
