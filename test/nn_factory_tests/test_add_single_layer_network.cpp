//
// Created by kfedrick on 6/19/19.
//

#include "test_nn_factory.h"

using flexnnet::PureLin;
using flexnnet::TanSig;
using flexnnet::RadBas;
using flexnnet::SoftMax;
using flexnnet::LogSig;

TEST_F(TestBasicNNFactory, SinglePureLin)
{
   BasicNeuralNetFactory factory;

   const PureLin::Parameters PARAMS = {.gain=1.631};
   std::shared_ptr<PureLin> layer_ptr = factory
      .add_layer<PureLin>(SINGLE_SZ, SINGLE_PURELIN_ID, SINGLE_LTYPE, PARAMS);

   ASSERT_EQ(SINGLE_SZ, layer_ptr->size());
   ASSERT_EQ(SINGLE_PURELIN_ID, layer_ptr->name());
   ASSERT_EQ(SINGLE_IS_OUTPUT, layer_ptr->is_output_layer());
   ASSERT_EQ(PARAMS.gain, layer_ptr->get_gain());
}

TEST_F(TestBasicNNFactory, SingleLogSig)
{
   BasicNeuralNetFactory factory;

   const LogSig::Parameters PARAMS = {.gain=1.631};
   std::shared_ptr<LogSig> layer_ptr = factory.add_layer<LogSig>(SINGLE_SZ, SINGLE_LOGSIG_ID, SINGLE_LTYPE, PARAMS);

   ASSERT_EQ(SINGLE_SZ, layer_ptr->size());
   ASSERT_EQ(SINGLE_LOGSIG_ID, layer_ptr->name());
   ASSERT_EQ(SINGLE_IS_OUTPUT, layer_ptr->is_output_layer());
   ASSERT_EQ(PARAMS.gain, layer_ptr->get_gain());
}

TEST_F(TestBasicNNFactory, SingleTanSig)
{
   BasicNeuralNetFactory factory;

   const TanSig::Parameters PARAMS = {.gain=1.631};
   std::shared_ptr<TanSig> layer_ptr = factory
      .add_layer<TanSig>(SINGLE_SZ, SINGLE_TANSIG_ID, SINGLE_LTYPE, PARAMS);

   ASSERT_EQ(SINGLE_SZ, layer_ptr->size());
   ASSERT_EQ(SINGLE_TANSIG_ID, layer_ptr->name());
   ASSERT_EQ(SINGLE_IS_OUTPUT, layer_ptr->is_output_layer());
   ASSERT_EQ(PARAMS.gain, layer_ptr->get_gain());
}

TEST_F(TestBasicNNFactory, SingleRadBas)
{
   BasicNeuralNetFactory factory;

   RadBas::Parameters PARAMS = {.rescaled_flag=true};
   std::shared_ptr<RadBas> layer_ptr = factory
      .add_layer<RadBas>(SINGLE_SZ, SINGLE_RADBAS_ID, SINGLE_LTYPE, PARAMS);

   ASSERT_EQ(SINGLE_SZ, layer_ptr->size());
   ASSERT_EQ(SINGLE_RADBAS_ID, layer_ptr->name());
   ASSERT_EQ(SINGLE_IS_OUTPUT, layer_ptr->is_output_layer());
   ASSERT_EQ(PARAMS.rescaled_flag, layer_ptr->is_rescaled());
}

TEST_F(TestBasicNNFactory, SingleSoftMax)
{
   BasicNeuralNetFactory factory;

   SoftMax::Parameters PARAMS = {.gain=1.631, .rescaled_flag=true};
   std::shared_ptr<SoftMax> layer_ptr = factory
      .add_layer<SoftMax>(SINGLE_SZ, SINGLE_SOFTMAX_ID, SINGLE_LTYPE, PARAMS);

   ASSERT_EQ(SINGLE_SZ, layer_ptr->size());
   ASSERT_EQ(SINGLE_SOFTMAX_ID, layer_ptr->name());
   ASSERT_EQ(SINGLE_IS_OUTPUT, layer_ptr->is_output_layer());
   ASSERT_EQ(PARAMS.rescaled_flag, layer_ptr->is_rescaled());
}