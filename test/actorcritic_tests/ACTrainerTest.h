//
// Created by kfedrick on 5/30/21.
//

#ifndef FLEX_NEURALNET_ACTRAINERTEST_H_
#define FLEX_NEURALNET_ACTRAINERTEST_H_

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
#include <Environment.h>
#include <Reinforcement.h>
#include <picojson.h>
#include <fstream>
#include <ActorCriticDeepRLAlgo.h>
#include <ActorCriticC2GFitnessFunc.h>
#include <ActorCriticFinalCostFitnessFunc.h>
#include <ConstantLearningRate.h>

#include "TestAction.h"

#include "ACTestFixture.h"
#include <ActionSet.h>
#include "DerbySim.h"
#include "DerbySim0.h"
#include "DerbySim2.h"
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

struct TrainerTestCase
{
   Array2D<double> actor_hweights;
   Array2D<double> actor_oweights;
   Array2D<double> critic_hweights;
   Array2D<double> critic_oweights;
};

template<typename T>
class ACTrainerTestFixture : public ACTestFixture<T>
{
public:

   virtual void
   SetUp()
   {}

   virtual void
   TearDown()
   {}

protected:
   std::vector<TrainerTestCase>& read_trainer_test_cases(const std::string& _fname);

   std::valarray<double> read_vector(const picojson::value& _v);

   Array2D<double> read_array2d(const picojson::value& _v);

private:
   std::vector<TrainerTestCase> trainer_test_cases;
};
TYPED_TEST_CASE_P (ACTrainerTestFixture);


template<typename T>
std::vector<TrainerTestCase>& ACTrainerTestFixture<T>::read_trainer_test_cases(const std::string& _fname)
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

   trainer_test_cases.clear();

   if (v.is<picojson::array>())
   {
      const picojson::array& root_arr = v.get<picojson::array>();

      TrainerTestCase tcase;
      for (int casendx = 0; casendx < root_arr.size(); casendx++)
      {
         picojson::value arr2d;
         picojson::array picoarr;

         // get the object containing the key/value pairs
         const picojson::object& o = root_arr[casendx].get<picojson::object>();

         // Initial NN weights
         arr2d = o.at("actor_hidden_weights");
         tcase.actor_hweights.set(read_array2d(arr2d));

         arr2d = o.at("actor_action_weights");
         tcase.actor_oweights.set(read_array2d(arr2d));

         arr2d = o.at("critic_hidden_weights");
         tcase.critic_hweights.set(read_array2d(arr2d));

         arr2d = o.at("critic_R_weights");
         tcase.critic_oweights.set(read_array2d(arr2d));

         trainer_test_cases.push_back(tcase);
      }
   }

   return trainer_test_cases;
}

template<typename T>
std::valarray<double> ACTrainerTestFixture<T>::read_vector(const picojson::value& _v)
{
   std::valarray<double> valar;

   const picojson::array& arr = _v.get<picojson::array>();
   valar.resize(arr.size());
   for (int i = 0; i < arr.size(); i++)
      valar[i] = arr[i].get<double>();

   return valar;
}


template<typename T>
Array2D<double> ACTrainerTestFixture<T>::read_array2d(const picojson::value& _arr2d)
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

TYPED_TEST_P(ACTrainerTestFixture, TestFinalACCritic)
{
   std::string layer_type_id = ACTrainerTestFixture<TypeParam>::get_typeid();

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   size_t actor_osize = 1;
   if (_id=="softmax")
      actor_osize = 2;

   std::cout << "----- AC Trainer<" << _id << "> Train Critic -----\n" << std::flush;
   GTEST_SKIP();

   RawFeatureSet<14> isample({"state"});
   ActionSet<SteeringActionFeature> action;
   DerbyStateAction0 stateaction({"state","action"});

   NeuralNet<RawFeatureSet<14>, ActionSet<SteeringActionFeature>> act = this->template create_actor<RawFeatureSet<14>, ActionSet<SteeringActionFeature>>(isample, actor_osize);
   NeuralNet<DerbyStateAction0, Reinforcement<1>> crit = this->template create_critic<DerbyStateAction0, Reinforcement<1>>(stateaction);

   BaseActorCriticNetwork<RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1> acnn(act,crit);

   //BaseActorCriticNetwork<RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1> acnn = this->template create_actorcritic<RawFeatureSet<23>, ActionSet<SteeringActionFeature>, DerbyStateAction, 1>(isample, stateaction, actor_osize);
   flexnnet::NeuralNet<RawFeatureSet<14>, ActionSet<SteeringActionFeature>>& actor = acnn.get_actor();
   flexnnet::NeuralNet<DerbyStateAction0, Reinforcement<1>>& critic = acnn.get_critic();

   //flexnnet::ActorCriticDeepRLAlgo<flexnnet::RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1, BaseActorCriticNetwork, DerbySim0, flexnnet::ActorCriticFinalFitnessFunc,
   //                                flexnnet::ConstantLearningRate> trainer(acnn);
   flexnnet::ActorCriticDeepRLAlgo<flexnnet::RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1, DerbySim0, flexnnet::ActorCriticFinalCostFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer;

   DerbySim0<RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1> derby_sim;

   RawFeatureSet<14> in({"state"});
   std::tuple<ActionSet<SteeringActionFeature>, flexnnet::Reinforcement<1>> nnout;

   std::string sample_fname = _id + "_ac_trainer0.json";

   const std::vector<TrainerTestCase>& test_cases = this->read_trainer_test_cases(sample_fname);
   for (auto& tcase : test_cases)
   {
      actor.set_weights("actor-hidden", tcase.actor_hweights);
      actor.set_weights("action", tcase.actor_oweights);
      critic.set_weights("critic-hidden", tcase.critic_hweights);
      critic.set_weights("R", tcase.critic_oweights);

      //trainer.set_td_mode(flexnnet::TDTrainerConfig::FINAL_COST);

      trainer.set_training_runs(1);
      trainer.set_max_epochs(2500);
      trainer.set_batch_mode(30);
      trainer.set_lambda(0.3);

      flexnnet::LayerWeights iahw = actor.get_weights("actor-hidden");
      flexnnet::LayerWeights iaow = actor.get_weights("action");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial actor hidden weights", iahw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial actor output weights", iaow.const_weights_ref);

      flexnnet::LayerWeights ichw = critic.get_weights("critic-hidden");
      flexnnet::LayerWeights icow = critic.get_weights("R");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial critic hidden weights", ichw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial critic output weights", icow.const_weights_ref);

      acnn.set_random_action(true);
      trainer.train(acnn, derby_sim);

      flexnnet::LayerWeights ahw = actor.get_weights("actor-hidden");
      flexnnet::LayerWeights aow = actor.get_weights("action");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor hidden weights", ahw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor output weights", aow.const_weights_ref);

      flexnnet::LayerWeights chw = critic.get_weights("critic-hidden");
      flexnnet::LayerWeights cow = critic.get_weights("R");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic hidden weights", chw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic output weights", cow.const_weights_ref);

      // Static evaluation of critic
      RawFeatureSet<14> test_state;
      std::valarray<double> test_statev(14);
      std::valarray<double> Rvals(14);

      int y_pos = 1;
      const int SAMPLES = 50;
      for (int x_pos=1; x_pos<11; x_pos++)
      {
         double mean_R = 0;
         std::tuple<ActionSet<SteeringActionFeature>, Reinforcement<1>> ret;
         derby_sim.set(x_pos, 0);
         test_state = derby_sim.state();
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("INIT test statev", std::get<0>(test_state.get_features()).get_encoding(), 1) << "\n";

         for (int sample_no = 0; sample_no < SAMPLES; sample_no++)
         {
            ret = acnn.activate(test_state);

            //mean_R += std::get<1>(ret)[0] / SAMPLES;
            mean_R += std::get<0>(std::get<1>(ret).get_features()).get_encoding()[0] / SAMPLES;
         }

         Rvals[x_pos] = mean_R;
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("test statev", std::get<0>(test_state.get_features()).get_encoding(), 1) << "\n";
         std::cout << "R value = " << std::get<0>(std::get<1>(ret).get_features()).get_encoding()[0] << std::setprecision(10) << "\n";
         std::cout << "R value = " << Rvals[x_pos] << std::setprecision(10) << "\n";


         std::cout << "-------------------------------------------------------\n";
      }

      std::cout << CommonTestFixtureFunctions::prettyPrintVector("R values", Rvals, 5) << "\n";
   }
}

TYPED_TEST_P(ACTrainerTestFixture, TestFinalAC)
{
   std::string layer_type_id = ACTrainerTestFixture<TypeParam>::get_typeid();

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   size_t actor_osize = 1;
   if (_id=="softmax")
      actor_osize = 2;

   std::cout << "----- AC Trainer<" << _id << "> Train AC Final -----\n" << std::flush;
   //GTEST_SKIP();

   RawFeatureSet<14> isample({"state"});
   ActionSet<SteeringActionFeature> action;
   DerbyStateAction0 stateaction({"state","action"});

   NeuralNet<RawFeatureSet<14>, ActionSet<SteeringActionFeature>> act = this->template create_actor<RawFeatureSet<14>, ActionSet<SteeringActionFeature>>(isample, actor_osize);
   NeuralNet<DerbyStateAction0, Reinforcement<1>> crit = this->template create_critic<DerbyStateAction0, Reinforcement<1>>(stateaction);

   BaseActorCriticNetwork<RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1> acnn(act,crit);

   //BaseActorCriticNetwork<RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1> acnn = this->template create_actorcritic<RawFeatureSet<23>, ActionSet<SteeringActionFeature>, DerbyStateAction, 1>(isample, stateaction, actor_osize);
   flexnnet::NeuralNet<RawFeatureSet<14>, ActionSet<SteeringActionFeature>>& actor = acnn.get_actor();
   flexnnet::NeuralNet<DerbyStateAction0, Reinforcement<1>>& critic = acnn.get_critic();

   //flexnnet::ActorCriticDeepRLAlgo<flexnnet::RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1, BaseActorCriticNetwork, DerbySim0, flexnnet::ActorCriticFinalFitnessFunc,
   //                                flexnnet::ConstantLearningRate> trainer(acnn);
   flexnnet::ActorCriticDeepRLAlgo<flexnnet::RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1, DerbySim0, flexnnet::ActorCriticFinalCostFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer;

   DerbySim0<RawFeatureSet<14>, ActionSet<SteeringActionFeature>, 1> derby_sim;

   RawFeatureSet<14> in({"state"});
   std::tuple<ActionSet<SteeringActionFeature>, flexnnet::Reinforcement<1>> nnout;

   std::string sample_fname = _id + "_ac_trainer0.json";

   const std::vector<TrainerTestCase>& test_cases = this->read_trainer_test_cases(sample_fname);
   for (auto& tcase : test_cases)
   {
      actor.set_weights("actor-hidden", tcase.actor_hweights);
      actor.set_weights("action", tcase.actor_oweights);
      critic.set_weights("critic-hidden", tcase.critic_hweights);
      critic.set_weights("R", tcase.critic_oweights);

      //trainer.set_td_mode(flexnnet::TDTrainerConfig::FINAL_COST);

      trainer.set_training_runs(1);
      trainer.set_max_epochs(1);
      trainer.set_batch_mode(30);
      trainer.set_lambda(0.3);

      flexnnet::LayerWeights iahw = actor.get_weights("actor-hidden");
      flexnnet::LayerWeights iaow = actor.get_weights("action");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial actor hidden weights", iahw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial actor output weights", iaow.const_weights_ref);

      flexnnet::LayerWeights ichw = critic.get_weights("critic-hidden");
      flexnnet::LayerWeights icow = critic.get_weights("R");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial critic hidden weights", ichw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial critic output weights", icow.const_weights_ref);

      acnn.set_random_action(false);
      trainer.train(acnn, derby_sim);

      flexnnet::LayerWeights ahw = actor.get_weights("actor-hidden");
      flexnnet::LayerWeights aow = actor.get_weights("action");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor hidden weights", ahw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor output weights", aow.const_weights_ref);

      flexnnet::LayerWeights chw = critic.get_weights("critic-hidden");
      flexnnet::LayerWeights cow = critic.get_weights("R");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic hidden weights", chw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic output weights", cow.const_weights_ref);

      // Static evaluation of critic
      RawFeatureSet<14> test_state;
      std::valarray<double> test_statev(14);
      std::valarray<double> Rvals(14);

      int y_pos = 1;
      const int SAMPLES = 50;
      for (int x_pos=1; x_pos<11; x_pos++)
      {
         double mean_R = 0;
         std::tuple<ActionSet<SteeringActionFeature>, Reinforcement<1>> ret;
         derby_sim.set(x_pos, 0);
         test_state = derby_sim.state();
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("INIT test statev", std::get<0>(test_state.get_features()).get_encoding(), 1) << "\n";

         for (int sample_no = 0; sample_no < SAMPLES; sample_no++)
         {
            ret = acnn.activate(test_state);

            //mean_R += std::get<1>(ret)[0] / SAMPLES;
            mean_R += std::get<0>(std::get<1>(ret).get_features()).get_encoding()[0] / SAMPLES;
         }

         Rvals[x_pos] = mean_R;
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("test statev", std::get<0>(test_state.get_features()).get_encoding(), 1) << "\n";
         std::cout << "R value = " << std::get<0>(std::get<1>(ret).get_features()).get_encoding()[0] << std::setprecision(10) << "\n";
         std::cout << "R value = " << Rvals[x_pos] << std::setprecision(10) << "\n";
         std::cout << "Action = " << std::get<0>(std::get<0>(ret).get_features()).get_encoding()[0] << std::setprecision(10) << "\n";

         std::cout << "Decoded Action: ";
         switch (std::get<0>(std::get<0>(ret).get_features()).get_action())
         {
            case SteeringActionFeature::ActionEnum::Left:
               std::cout << "Left\n";
               break;
            case SteeringActionFeature::ActionEnum::Right:
               std::cout << "Right\n";
               break;
            default:
               std::cout << "neither\n";
         };

         std::cout << "-------------------------------------------------------\n";
      }

      std::cout << CommonTestFixtureFunctions::prettyPrintVector("R values", Rvals, 5) << "\n";
   }
}
/*
TYPED_TEST_P(ACTrainerTestFixture, TrainerTest1)
{
   std::string layer_type_id = ACTrainerTestFixture<TypeParam>::get_typeid();

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   size_t actor_osize = 1;
   if (_id=="softmax")
      actor_osize = 2;

   std::cout << "----- AC Trainer<" << _id << "> Test 1 -----\n" << std::flush;
   GTEST_SKIP();

   RawFeatureSet<23> isample;
   ActionSet<SteeringActionFeature> action;
   DerbyStateAction stateaction;

   NeuralNet<RawFeatureSet<23>, ActionSet<SteeringActionFeature>> act = this->template create_actor<RawFeatureSet<23>, ActionSet<SteeringActionFeature>>(isample, actor_osize);
   NeuralNet<DerbyStateAction, Reinforcement<1>> crit = this->template create_critic<DerbyStateAction, Reinforcement<1>>(stateaction);

   BaseActorCriticNetwork<RawFeatureSet<23>, ActionSet<SteeringActionFeature>, 1> acnn(act,crit);

   //BaseActorCriticNetwork<RawFeatureSet<23>, ActionSet<SteeringActionFeature>, 1> acnn = this->template create_actorcritic<RawFeatureSet<23>, ActionSet<SteeringActionFeature>, DerbyStateAction, 1>(isample, stateaction, actor_osize);
   flexnnet::NeuralNet<RawFeatureSet<23>, ActionSet<SteeringActionFeature>>& actor = acnn.get_actor();
   flexnnet::NeuralNet<DerbyStateAction, Reinforcement<1>>& critic = acnn.get_critic();

   //flexnnet::ActorCriticDeepRLAlgo<flexnnet::RawFeatureSet<23>, ActionSet<SteeringActionFeature>, 1, BaseActorCriticNetwork, DerbySim, flexnnet::ActorCriticFinalFitnessFunc,
   //                                flexnnet::ConstantLearningRate> trainer(acnn);
   flexnnet::ActorCriticDeepRLAlgo<flexnnet::RawFeatureSet<23>, ActionSet<SteeringActionFeature>, 1, BaseActorCriticNetwork, DerbySim, flexnnet::TDFinalCostFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer(acnn);

   DerbySim<RawFeatureSet<23>, ActionSet<SteeringActionFeature>, 1> derby_sim;

   RawFeatureSet<23> in;
   std::tuple<ActionSet<SteeringActionFeature>, flexnnet::Reinforcement<1>> nnout;

   std::string sample_fname = _id + "_ac_trainer.json";

   const std::vector<TrainerTestCase>& test_cases = this->read_trainer_test_cases(sample_fname);
   for (auto& tcase : test_cases)
   {
      actor.set_weights("actor-hidden", tcase.actor_hweights);
      actor.set_weights("F1", tcase.actor_oweights);
      critic.set_weights("critic-hidden", tcase.critic_hweights);
      critic.set_weights("R", tcase.critic_oweights);

      trainer.set_training_runs(1);
      trainer.set_max_epochs(1500);
      trainer.set_batch_mode(1);
      trainer.set_lambda(0.8);

      flexnnet::LayerWeights iahw = actor.get_weights("actor-hidden");
      flexnnet::LayerWeights iaow = actor.get_weights("F1");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial actor hidden weights", iahw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial actor output weights", iaow.const_weights_ref);

      flexnnet::LayerWeights ichw = critic.get_weights("critic-hidden");
      flexnnet::LayerWeights icow = critic.get_weights("R");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial critic hidden weights", ichw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial critic output weights", icow.const_weights_ref);

      trainer.train(derby_sim);

      flexnnet::LayerWeights ahw = actor.get_weights("actor-hidden");
      flexnnet::LayerWeights aow = actor.get_weights("F1");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor hidden weights", ahw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor output weights", aow.const_weights_ref);

      flexnnet::LayerWeights chw = critic.get_weights("critic-hidden");
      flexnnet::LayerWeights cow = critic.get_weights("R");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic hidden weights", chw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic output weights", cow.const_weights_ref);

      // Static evaluation of critic
      RawFeatureSet<23> test_state;
      std::valarray<double> test_statev(23);
      std::valarray<double> Rvals(23);

      int y_pos = 8;
      const int SAMPLES = 50;
      for (int x_pos=1; x_pos<11; x_pos++)
      {
         double mean_R = 0;
         std::tuple<ActionSet<SteeringActionFeature>, Reinforcement<1>> ret;
         derby_sim.set(x_pos, y_pos);
         test_state = derby_sim.state();
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("INIT test statev", std::get<0>(test_state.get_features()).get_encoding(), 1) << "\n";

         for (int sample_no = 0; sample_no < SAMPLES; sample_no++)
         {
            ret = acnn.activate(test_state);

            //mean_R += std::get<1>(ret)[0] / SAMPLES;
            mean_R += std::get<0>(std::get<1>(ret).get_features()).get_encoding()[0] / SAMPLES;
         }

         Rvals[x_pos] = mean_R;
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("test statev", std::get<0>(test_state.get_features()).get_encoding(), 1) << "\n";
         std::cout << "R value = " << std::get<0>(std::get<1>(ret).get_features()).get_encoding()[0] << std::setprecision(10) << "\n";
         std::cout << "R value = " << Rvals[x_pos] << std::setprecision(10) << "\n";

         std::cout << "-------------------------------------------------------\n";
      }

      std::cout << CommonTestFixtureFunctions::prettyPrintVector("R values", Rvals, 5) << "\n";
   }
}

TYPED_TEST_P(ACTrainerTestFixture, TrainerTest2)
{
   std::string layer_type_id = ACTrainerTestFixture<TypeParam>::get_typeid();

   // Get lower case parameterized type string
   std::string _id = layer_type_id;
   std::transform(_id.begin(), _id.end(), _id.begin(), ::tolower);

   size_t actor_osize = 1;
   if (_id=="softmax")
      actor_osize = 2;

   std::cout << "----- AC Trainer<" << _id << "> Test 2 -----\n" << std::flush;
   //GTEST_SKIP();

   DerbySim2<RawFeatureSet<(10+2)*(10+1)>, ActionSet<SteeringActionFeature>, 1> derby_sim;
   RawFeatureSet<(10+2)*(10+1)> isample;
   ActionSet<SteeringActionFeature> action;
   DerbyStateAction2 stateaction;

   NeuralNet<RawFeatureSet<(10+2)*(10+1)>, ActionSet<SteeringActionFeature>> act = this->template create_actor<RawFeatureSet<(10+2)*(10+1)>, ActionSet<SteeringActionFeature>>(isample, actor_osize);
   NeuralNet<DerbyStateAction2, Reinforcement<1>> crit = this->template create_critic<DerbyStateAction2, Reinforcement<1>>(stateaction);

   BaseActorCriticNetwork<RawFeatureSet<(10+2)*(10+1)>, ActionSet<SteeringActionFeature>, 1> acnn(act,crit);

   //BaseActorCriticNetwork<RawFeatureSet<23>, ActionSet<SteeringActionFeature>, 1> acnn = this->template create_actorcritic<RawFeatureSet<23>, ActionSet<SteeringActionFeature>, DerbyStateAction, 1>(isample, stateaction, actor_osize);
   flexnnet::NeuralNet<RawFeatureSet<(10+2)*(10+1)>, ActionSet<SteeringActionFeature>>& actor = acnn.get_actor();
   flexnnet::NeuralNet<DerbyStateAction2, Reinforcement<1>>& critic = acnn.get_critic();

   //flexnnet::ActorCriticDeepRLAlgo<RawFeatureSet<(10+2)*(10+1)>, ActionSet<SteeringActionFeature>, 1, BaseActorCriticNetwork, DerbySim2, flexnnet::ActorCriticFinalFitnessFunc,
   //                                flexnnet::ConstantLearningRate> trainer(acnn);
   flexnnet::ActorCriticDeepRLAlgo<RawFeatureSet<(10+2)*(10+1)>, ActionSet<SteeringActionFeature>, 1, BaseActorCriticNetwork, DerbySim2, flexnnet::TDFinalCostFitnessFunc,
                                   flexnnet::ConstantLearningRate> trainer(acnn);

   RawFeatureSet<(10+2)*(10+1)> in;
   std::tuple<ActionSet<SteeringActionFeature>, flexnnet::Reinforcement<1>> nnout;

   std::cout << "here\n" << std::flush;

   std::string sample_fname = _id + "_ac_trainer2.json";

   const std::vector<TrainerTestCase>& test_cases = this->read_trainer_test_cases(sample_fname);

   std::cout << "and here\n" << std::flush;

   for (auto& tcase : test_cases)
   {
      actor.set_weights("actor-hidden", tcase.actor_hweights);
      actor.set_weights("F1", tcase.actor_oweights);
      critic.set_weights("critic-hidden", tcase.critic_hweights);
      critic.set_weights("R", tcase.critic_oweights);

      std::cout << "what about here\n" << std::flush;

      //trainer.set_td_mode(flexnnet::TDTrainerConfig::FINAL_COST);
      trainer.set_training_runs(1);
      trainer.set_max_epochs(40);
      trainer.set_batch_mode(1);
      trainer.set_lambda(0.4);

      flexnnet::LayerWeights iahw = actor.get_weights("actor-hidden");
      flexnnet::LayerWeights iaow = actor.get_weights("F1");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial actor hidden weights", iahw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial actor output weights", iaow.const_weights_ref);

      flexnnet::LayerWeights ichw = critic.get_weights("critic-hidden");
      flexnnet::LayerWeights icow = critic.get_weights("R");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial critic hidden weights", ichw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("initial critic output weights", icow.const_weights_ref);

      trainer.train(derby_sim);

      flexnnet::LayerWeights ahw = actor.get_weights("actor-hidden");
      flexnnet::LayerWeights aow = actor.get_weights("F1");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor hidden weights", ahw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("actor output weights", aow.const_weights_ref);

      flexnnet::LayerWeights chw = critic.get_weights("critic-hidden");
      flexnnet::LayerWeights cow = critic.get_weights("R");

      std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic hidden weights", chw.const_weights_ref);
      std::cout << CommonTestFixtureFunctions::prettyPrintArray("critic output weights", cow.const_weights_ref);

      // Static evaluation of critic
      RawFeatureSet<(10+2)*(10+1)> test_state;
      std::valarray<double> test_statev((10+2)*(10+1));
      std::valarray<double> Rvals(10);

      const int SAMPLES = 500;
      int y_pos = 3;
      for (int x_pos=1; x_pos<11; x_pos++)
      {
         double mean_R = 0;
         std::tuple<ActionSet<SteeringActionFeature>, Reinforcement<1>> ret;
         derby_sim.set(x_pos, y_pos);
         test_state = derby_sim.state();
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("INIT ds2 test statev", std::get<0>(test_state.get_features()).get_encoding(), 1) << "\n";

         for (int sample_no = 0; sample_no < SAMPLES; sample_no++)
         {
            ret = acnn.activate(test_state);

            //mean_R += std::get<1>(ret)[0] / SAMPLES;
            mean_R += std::get<0>(std::get<1>(ret).get_features()).get_encoding()[0] / SAMPLES;
         }

         Rvals[x_pos] = mean_R;
         std::cout << CommonTestFixtureFunctions::prettyPrintVector("test statev", std::get<0>(test_state.get_features()).get_encoding(), 1) << "\n";
         std::cout << "R value = " << std::get<0>(std::get<1>(ret).get_features()).get_encoding()[0] << std::setprecision(10) << "\n";
         std::cout << "R value = " << Rvals[x_pos] << std::setprecision(10) << "\n";

         std::cout << "-------------------------------------------------------\n";
      }

      std::cout << CommonTestFixtureFunctions::prettyPrintVector("R values", Rvals, 5) << "\n";

   }
}
*/

REGISTER_TYPED_TEST_CASE_P
(ACTrainerTestFixture, TestFinalACCritic, TestFinalAC/*, TrainerTest1, TrainerTest2*/);
INSTANTIATE_TYPED_TEST_CASE_P
(My, ACTrainerTestFixture, MyTypes);

#endif // FLEX_NEURALNET_ACTRAINERTEST_H_
