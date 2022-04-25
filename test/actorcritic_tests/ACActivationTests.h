//
// Created by kfedrick on 5/21/21.
//

#ifndef _ACACTIVATIONTESTS_H_
#define _ACACTIVATIONTESTS_H_

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

#include "ACTestFixture.h"

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

struct ActivationTestCase
{
   Array2D<double> actor_hweights;
   Array2D<double> actor_oweights;
   Array2D<double> critic_hweights;
   Array2D<double> critic_oweights;

   std::valarray<double> state;
   std::valarray<double> tgt_action_enc;
   std::valarray<double> tgt_reinforcement;
};

struct BackpropTestCase
{
   Array2D<double> actor_hweights;
   Array2D<double> actor_oweights;
   Array2D<double> critic_hweights;
   Array2D<double> critic_oweights;

   std::valarray<double> state;
   std::valarray<double> tgt_action_enc;
   std::valarray<double> tgt_reinforcement;

   std::valarray<double> error_gradient;
   Array2D<double> actor_hdEdw;
   Array2D<double> actor_odEdw;
   Array2D<double> critic_hdEdw;
   Array2D<double> critic_odEdw;
   std::valarray<double> critic_dEdx0;
   std::valarray<double> critic_dEdx1;

};

template<typename T>
class ACActivationTestFixture : public ACTestFixture<T>
{
public:

   virtual void
   SetUp()
   {}

   virtual void
   TearDown()
   {}

protected:
   std::vector<ActivationTestCase>& read_activation_test_cases(const std::string& _fname);
   std::vector<BackpropTestCase>& read_backprop_test_cases(const std::string& _fname);

   std::valarray<double> read_vector(const picojson::value& _v);

   Array2D<double> read_array2d(const picojson::value& _v);

private:
   std::vector<ActivationTestCase> activation_test_cases;
   std::vector<BackpropTestCase> backprop_test_cases;
};
TYPED_TEST_CASE_P (ACActivationTestFixture);


template<typename T>
std::vector<ActivationTestCase>& ACActivationTestFixture<T>::read_activation_test_cases(const std::string& _fname)
{
   picojson::value v;
   std::string err;

   std::ifstream ifs(_fname);
   if (!ifs.good())
   {
      std::ostringstream err_str;
      err_str << "File not found (" << _fname << ")\n";
      throw std::invalid_argument(err_str.str());
   }

   std::string json(std::istreambuf_iterator<char>(ifs), {});
   err = picojson::parse(v, json);
   ifs.close();

   activation_test_cases.clear();

   if (v.is<picojson::array>())
   {
      const picojson::array& root_arr = v.get<picojson::array>();

      ActivationTestCase tcase;
      for (int casendx = 0; casendx < root_arr.size(); casendx++)
      {
         picojson::value arr2d;
         picojson::array picoarr;

         // get the object containing the key/value pairs
         const picojson::object& o = root_arr[casendx].get<picojson::object>();

         const picojson::value& statev = o.at("state");
         tcase.state = read_vector(statev);

         const picojson::value& tgt_R = o.at("target_reinforcement");
         tcase.tgt_reinforcement = read_vector(tgt_R);

         const picojson::value& tgt_action = o.at("target_action");
         tcase.tgt_action_enc = read_vector(tgt_action);

         arr2d = o.at("actor_hidden_weights");
         tcase.actor_hweights.set(read_array2d(arr2d));

         arr2d = o.at("actor_output_weights");
         tcase.actor_oweights.set(read_array2d(arr2d));

         arr2d = o.at("critic_hidden_weights");
         tcase.critic_hweights.set(read_array2d(arr2d));

         arr2d = o.at("critic_output_weights");
         tcase.critic_oweights.set(read_array2d(arr2d));

         activation_test_cases.push_back(tcase);
      }
   }

   return activation_test_cases;
}

template<typename T>
std::vector<BackpropTestCase>& ACActivationTestFixture<T>::read_backprop_test_cases(const std::string& _fname)
{
   picojson::value v;
   std::string err;

   std::ifstream ifs(_fname);
   if (!ifs.good())
   {
      std::ostringstream err_str;
      err_str << "File not found (" << _fname << ")\n";
      throw std::invalid_argument(err_str.str());
   }

   std::string json(std::istreambuf_iterator<char>(ifs), {});
   err = picojson::parse(v, json);
   ifs.close();

   activation_test_cases.clear();

   if (v.is<picojson::array>())
   {
      const picojson::array& root_arr = v.get<picojson::array>();

      BackpropTestCase tcase;
      for (int casendx = 0; casendx < root_arr.size(); casendx++)
      {
         picojson::value arr2d;
         picojson::array picoarr;

         // get the object containing the key/value pairs
         const picojson::object& o = root_arr[casendx].get<picojson::object>();

         const picojson::value& statev = o.at("state");
         tcase.state = read_vector(statev);

         const picojson::value& tgt_R = o.at("target_reinforcement");
         tcase.tgt_reinforcement = read_vector(tgt_R);

         const picojson::value& tgt_action = o.at("target_action");
         tcase.tgt_action_enc = read_vector(tgt_action);

         const picojson::value& egradient = o.at("error_gradient");
         tcase.error_gradient = read_vector(egradient);

         // Initial NN weights
         arr2d = o.at("actor_hidden_weights");
         tcase.actor_hweights.set(read_array2d(arr2d));

         arr2d = o.at("actor_output_weights");
         tcase.actor_oweights.set(read_array2d(arr2d));

         arr2d = o.at("critic_hidden_weights");
         tcase.critic_hweights.set(read_array2d(arr2d));

         arr2d = o.at("critic_output_weights");
         tcase.critic_oweights.set(read_array2d(arr2d));

         // dEdw
         arr2d = o.at("actor_hidden_dEdw");
         tcase.actor_hdEdw.set(read_array2d(arr2d));

         arr2d = o.at("actor_F1_dEdw");
         tcase.actor_odEdw.set(read_array2d(arr2d));

         arr2d = o.at("critic_hidden_dEdw");
         tcase.critic_hdEdw.set(read_array2d(arr2d));

         arr2d = o.at("critic_R_dEdw");
         tcase.critic_odEdw.set(read_array2d(arr2d));

         // dEdx
         const picojson::value& critic_dEdx0 = o.at("critic_hidden_F0_dEdx");
         tcase.critic_dEdx0 = read_vector(critic_dEdx0);

         const picojson::value& critic_dEdx1 = o.at("critic_hidden_F1_dEdx");
         tcase.critic_dEdx1 = read_vector(critic_dEdx1);

         backprop_test_cases.push_back(tcase);
      }
   }

   return backprop_test_cases;
}

template<typename T>
std::valarray<double> ACActivationTestFixture<T>::read_vector(const picojson::value& _v)
{
   std::valarray<double> valar;

   const picojson::array& arr = _v.get<picojson::array>();
   valar.resize(arr.size());
   for (int i = 0; i < arr.size(); i++)
      valar[i] = arr[i].get<double>();

   return valar;
}


template<typename T>
Array2D<double> ACActivationTestFixture<T>::read_array2d(const picojson::value& _arr2d)
{
   Array2D<double> darr;

   const picojson::array& amajor = _arr2d.get<picojson::array>();
   const picojson::array& aminor = amajor[0].get<picojson::array>();

   darr.resize(amajor.size(), aminor.size());
   for (int row = 0; row < amajor.size(); row++)
   {
      const picojson::array& aminor = amajor[row].get<picojson::array>();
      for (int col = 0; col < aminor.size(); col++)
         darr.at(row,col) = aminor[col].get<double>();
   }

   return darr;
}



TYPED_TEST_P(ACActivationTestFixture, ConstructorTest)
{
   std::string layer_type_id = ACActivationTestFixture<TypeParam>::get_typeid();

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   size_t actor_osize = 1;
   if (_id=="softmax")
      actor_osize = 2;

   std::cout << "----- AC Constructor<" << _id << "> Test -----\n" << std::flush;


   InputFeatures isample;
   TestAction action;
   StateAction stateaction;

   NeuralNet<InputFeatures, TestAction> actor = this->template create_actor<InputFeatures, TestAction>(isample, actor_osize);
   NeuralNet<StateAction, Reinforcement<1>> critic = this->template create_critic<StateAction, Reinforcement<1>>(stateaction);


   NeuralNet<InputFeatures, TestAction> tmpactor = this->template create_actor<InputFeatures, TestAction>(isample, actor_osize);
   NeuralNet<StateAction, Reinforcement<1>> tmpcritic = this->template create_critic<StateAction, Reinforcement<1>>(stateaction);

   BaseActorCriticNetwork<InputFeatures, TestAction, 1> acnn(tmpactor,tmpcritic);

   //BaseActorCriticNetwork<InputFeatures, TestAction, 1> acnn = this->template create_actorcritic<InputFeatures, TestAction, StateAction, 1>(isample, stateaction, actor_osize);
   std::cout << "----------------------------------------\n\n" << std::flush;
}


TYPED_TEST_P(ACActivationTestFixture, ActivationTest)
{
   std::string layer_type_id = ACActivationTestFixture<TypeParam>::get_typeid();

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   size_t actor_osize = 1;
   if (_id=="softmax")
      actor_osize = 2;

   std::cout << "-----  AC Activation<" << _id << "> Test  -----\n" << std::flush;

   InputFeatures isample;
   TestAction action;
   StateAction stateaction;

   NeuralNet<InputFeatures, TestAction> tmpactor = this->template create_actor<InputFeatures, TestAction>(isample, actor_osize);
   NeuralNet<StateAction, Reinforcement<1>> tmpcritic = this->template create_critic<StateAction, Reinforcement<1>>(stateaction);

   BaseActorCriticNetwork<InputFeatures, TestAction, 1> acnn(tmpactor,tmpcritic);

   //BaseActorCriticNetwork<InputFeatures, TestAction, 1> acnn = this->template create_actorcritic<InputFeatures, TestAction, StateAction, 1>(isample, stateaction, actor_osize);
   flexnnet::NeuralNet<InputFeatures, TestAction>& actor = acnn.get_actor();
   flexnnet::NeuralNet<StateAction, Reinforcement<1>>& critic = acnn.get_critic();

   InputFeatures in;
   std::tuple<TestAction, flexnnet::Reinforcement<1>> nnout;

   std::string sample_fname = _id + "_ac_activate.json";
   const std::vector<ActivationTestCase>& test_cases = this->read_activation_test_cases(sample_fname);

   for (auto& tcase : test_cases)
   {
      actor.set_weights("actor-hidden", tcase.actor_hweights);
      actor.set_weights("F1", tcase.actor_oweights);
      critic.set_weights("critic-hidden", tcase.critic_hweights);
      critic.set_weights("R", tcase.critic_oweights);

      in.decode({tcase.state});
      nnout = acnn.activate(in);

      //std::cout << this->template prettyPrintVector("in", tcase.state) << "\n\n";


      std::valarray<double> nnout_R = std::get<0>(std::get<1>(nnout).get_features()).get_encoding();
      std::valarray<double>
         nnout_A_enc = std::get<0>(std::get<0>(nnout).get_features()).get_encoding();



/*
      std::cout << this->template prettyPrintVector("nnout_R", nnout_R) << "\n\n";
      std::cout << this->template prettyPrintVector("nnout Action", nnout_A_enc) << "\n\n";
      std::cout << this->template prettyPrintVector("tgt R", tcase.tgt_reinforcement) << "\n\n";
      std::cout << this->template prettyPrintVector("tgt Action", tcase.tgt_action_enc) << "\n\n";
*/

      // Test AC reinforcement and action encoding against targets
      bool R_eq = CommonTestFixtureFunctions::valarray_double_near(nnout_R, tcase.tgt_reinforcement, 1e-5);
      bool A_eq = CommonTestFixtureFunctions::valarray_double_near(nnout_A_enc, tcase.tgt_action_enc, 1e-5);

      EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, nnout_R, tcase.tgt_reinforcement, 1e-5) << "ruh roh";
      EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, nnout_A_enc, tcase.tgt_action_enc, 1e-5) << "ruh roh";

      if (!R_eq || !A_eq)
      {
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("State", tcase.state) << "\n";
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("Reinforcement", nnout_R, 7) << "\n";
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("Action encoding", nnout_A_enc, 7) << "\n";

         std::cout << "Decoded Action: ";
         switch (std::get<0>(std::get<0>(nnout).get_features()).get_action())
         {
            case TestActionFeature::ActionEnum::Left:
               std::cout << "Left\n";
               break;
            case TestActionFeature::ActionEnum::Right:
               std::cout << "Right\n";
               break;
            default:
               std::cout << "neither\n";
         };


         flexnnet::LayerWeights ahw = actor.get_weights("actor-hidden");
         flexnnet::LayerWeights aow = actor.get_weights("F1");

         std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor hidden weights", ahw.const_weights_ref);
         std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor output weights", aow.const_weights_ref);

         flexnnet::LayerWeights chw = critic.get_weights("critic-hidden");
         flexnnet::LayerWeights cow = critic.get_weights("R");

         std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic hidden weights", chw.const_weights_ref);
         std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic output weights", cow.const_weights_ref);
      }

      std::cout << "\n----------------------------------------\n" << std::flush;

   }
   std::cout << "********************************************\n" << std::flush;
}



TYPED_TEST_P(ACActivationTestFixture, BackpropCriticTest)
{
   std::string layer_type_id = ACActivationTestFixture<TypeParam>::get_typeid();

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   size_t actor_osize = 1;
   if (_id=="softmax")
      actor_osize = 2;

   std::cout << "----- AC Backprop Critic<" << _id << "> Test -----\n" << std::flush;

   InputFeatures isample;
   TestAction action;
   StateAction stateaction;

   NeuralNet<InputFeatures, TestAction> tmpactor = this->template create_actor<InputFeatures, TestAction>(isample, actor_osize);
   NeuralNet<StateAction, Reinforcement<1>> tmpcritic = this->template create_critic<StateAction, Reinforcement<1>>(stateaction);

   BaseActorCriticNetwork<InputFeatures, TestAction, 1> acnn(tmpactor,tmpcritic);

   //BaseActorCriticNetwork<InputFeatures, TestAction, 1> acnn = this->template create_actorcritic<InputFeatures, TestAction, StateAction, 1>(isample, stateaction, actor_osize);
   flexnnet::NeuralNet<InputFeatures, TestAction>& actor = acnn.get_actor();
   flexnnet::NeuralNet<StateAction, Reinforcement<1>>& critic = acnn.get_critic();

   flexnnet::ValarrMap ones = acnn.get_critic().value_map();
   for (auto& it : ones)
      it.second = -1.0;

   InputFeatures in;
   std::tuple<TestAction, flexnnet::Reinforcement<1>> nnout;

   std::string sample_fname = _id + "_ac_backprop_critic.json";
   const std::vector<BackpropTestCase>& test_cases = this->read_backprop_test_cases(sample_fname);
   for (auto& tcase : test_cases)
   {
      actor.set_weights("actor-hidden", tcase.actor_hweights);
      actor.set_weights("F1", tcase.actor_oweights);
      critic.set_weights("critic-hidden", tcase.critic_hweights);
      critic.set_weights("R", tcase.critic_oweights);

      in.decode({tcase.state});
      nnout = acnn.activate(in);

      acnn.backprop_critic(ones);

      std::string label;
      const std::map<std::string, std::shared_ptr<flexnnet::NetworkLayer>>
         & critic_network_layers = critic.get_layers();
      const std::map<std::string, std::shared_ptr<flexnnet::NetworkLayer>>
         & actor_network_layers = actor.get_layers();

      EXPECT_PRED3(CommonTestFixtureFunctions::array_double_near, critic_network_layers.at("R")->dEdw(), tcase.critic_odEdw, 1e-6) << "Bad critic R dEdw";
      EXPECT_PRED3(CommonTestFixtureFunctions::array_double_near, critic_network_layers.at("critic-hidden")->dEdw(), tcase.critic_hdEdw, 1e-6) << "Bad critic hidden dEdw";
      EXPECT_PRED3(CommonTestFixtureFunctions::array_double_near, actor_network_layers.at("F1")->dEdw(), tcase.actor_odEdw, 1e-6) << "Bad actor F1 dEdw";
      EXPECT_PRED3(CommonTestFixtureFunctions::array_double_near, actor_network_layers.at("actor-hidden")->dEdw(), tcase.actor_hdEdw, 1e-6) << "Bad actor hidden dEdw";
      EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, critic_network_layers.at("critic-hidden")->dEdx().at("F0"), tcase.critic_dEdx0, 1e-6) << "Bad critic dEdx F0";
      EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, critic_network_layers.at("critic-hidden")->dEdx().at("F1"), tcase.critic_dEdx1, 1e-6) << "Bad critic dEdx F1";

      for (auto& layer : critic_network_layers)
      {
         const Array2D<double>& dEdw = layer.second->dEdw();
         label = "critic " + layer.first + " dEdw";
         std::cout << CommonTestFixtureFunctions::prettyPrintArray(label, dEdw, 7);

         const flexnnet::ValarrMap& dEdx = layer.second->dEdx();

         for (auto dx : dEdx)
         {
            label = "critic " + layer.first + "_" + dx.first + " dEdx";
            std::cout << CommonTestFixtureFunctions::prettyPrintVector(label, dEdx.at(dx.first), 7)
                      << "\n";
         }
      }


      for (auto& layer : actor_network_layers)
      {
         const Array2D<double>& dEdw = layer.second->dEdw();
         label = "actor " + layer.first + " dEdw";
         std::cout << CommonTestFixtureFunctions::prettyPrintArray(label, dEdw, 7);
      }
   }
   std::cout << "----------------------------------------\n\n" << std::flush;
}

TYPED_TEST_P(ACActivationTestFixture, BackpropActorTest)
{
   std::string layer_type_id = ACActivationTestFixture<TypeParam>::get_typeid();

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   size_t actor_osize = 1;
   if (_id=="softmax")
      actor_osize = 2;

   std::cout << "----- AC Backprop Actor<" << _id << "> Test -----\n" << std::flush;

   InputFeatures isample;
   TestAction action;
   StateAction stateaction;

   NeuralNet<InputFeatures, TestAction> tmpactor = this->template create_actor<InputFeatures, TestAction>(isample, actor_osize);
   NeuralNet<StateAction, Reinforcement<1>> tmpcritic = this->template create_critic<StateAction, Reinforcement<1>>(stateaction);

   BaseActorCriticNetwork<InputFeatures, TestAction, 1> acnn(tmpactor,tmpcritic);

   //BaseActorCriticNetwork<InputFeatures, TestAction, 1> acnn = this->template create_actorcritic<InputFeatures, TestAction, StateAction, 1>(isample, stateaction, actor_osize);
   flexnnet::NeuralNet<InputFeatures, TestAction>& actor = acnn.get_actor();
   flexnnet::NeuralNet<StateAction, Reinforcement<1>>& critic = acnn.get_critic();

   flexnnet::ValarrMap ones = acnn.get_critic().value_map();
   for (auto& it : ones)
      it.second = -1.0;

   InputFeatures in;
   std::tuple<TestAction, flexnnet::Reinforcement<1>> nnout;

   std::string sample_fname = _id + "_ac_backprop_actor.json";
   const std::vector<BackpropTestCase>& test_cases = this->read_backprop_test_cases(sample_fname);
   for (auto& tcase : test_cases)
   {
      actor.set_weights("actor-hidden", tcase.actor_hweights);
      actor.set_weights("F1", tcase.actor_oweights);
      critic.set_weights("critic-hidden", tcase.critic_hweights);
      critic.set_weights("R", tcase.critic_oweights);

      in.decode({tcase.state});
      acnn.activate(in);

      acnn.backprop_actor(ones);

      const std::map<std::string, std::shared_ptr<flexnnet::NetworkLayer>>
         & critic_network_layers = critic.get_layers();
      const std::map<std::string, std::shared_ptr<flexnnet::NetworkLayer>>
         & actor_network_layers = actor.get_layers();

      EXPECT_PRED3(CommonTestFixtureFunctions::array_double_near, critic_network_layers.at("R")->dEdw(), tcase.critic_odEdw, 1e-6) << "Bad critic R dEdw";
      EXPECT_PRED3(CommonTestFixtureFunctions::array_double_near, critic_network_layers.at("critic-hidden")->dEdw(), tcase.critic_hdEdw, 1e-6) << "Bad critic hidden dEdw";
      EXPECT_PRED3(CommonTestFixtureFunctions::array_double_near, actor_network_layers.at("F1")->dEdw(), tcase.actor_odEdw, 1e-6) << "Bad actor F1 dEdw";
      EXPECT_PRED3(CommonTestFixtureFunctions::array_double_near, actor_network_layers.at("actor-hidden")->dEdw(), tcase.actor_hdEdw, 1e-6) << "Bad actor hidden dEdw";
      EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, critic_network_layers.at("critic-hidden")->dEdx().at("F0"), tcase.critic_dEdx0, 1e-6) << "Bad critic dEdx F0";
      EXPECT_PRED3(CommonTestFixtureFunctions::vector_double_near, critic_network_layers.at("critic-hidden")->dEdx().at("F1"), tcase.critic_dEdx1, 1e-6) << "Bad critic dEdx F1";

      std::string label;

      for (auto& layer : critic_network_layers)
      {
         const Array2D<double>& dEdw = layer.second->dEdw();
         label = "critic " + layer.first + " dEdw";
         std::cout << CommonTestFixtureFunctions::prettyPrintArray(label, dEdw, 7);

         const flexnnet::ValarrMap& dEdx = layer.second->dEdx();

         for (auto dx : dEdx)
         {
            label = "critic " + layer.first + "_" + dx.first + " dEdx";
            std::cout << CommonTestFixtureFunctions::prettyPrintVector(label, dEdx.at(dx.first), 7)
                      << "\n";
         }
      }

      for (auto& layer : actor_network_layers)
      {
         const Array2D<double>& dEdw = layer.second->dEdw();
         label = "actor " + layer.first + " dEdw";
         std::cout << CommonTestFixtureFunctions::prettyPrintArray(label, dEdw, 7);
      }
   }
   std::cout << "----------------------------------------\n\n" << std::flush;
}


REGISTER_TYPED_TEST_CASE_P
(ACActivationTestFixture, ConstructorTest, ActivationTest, BackpropCriticTest, BackpropActorTest);
INSTANTIATE_TYPED_TEST_CASE_P
(My, ACActivationTestFixture, MyTypes);

#endif //_ACACTIVATIONTESTS_H_
