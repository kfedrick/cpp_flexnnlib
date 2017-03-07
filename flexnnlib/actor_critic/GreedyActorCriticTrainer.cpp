/*
 * GreedyActorCriticTrainer.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#include "GreedyActorCriticTrainer.h"

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace std;

namespace flex_neuralnet
{

const long GreedyActorCriticTrainer::default_max_epochs = 1;
const double GreedyActorCriticTrainer::default_learning_momentum = 0.0;
const double GreedyActorCriticTrainer::default_performance_goal = 0.0;
const double GreedyActorCriticTrainer::default_min_gradient = 0.0;
const int GreedyActorCriticTrainer::default_max_valid_fail = 10;
const bool GreedyActorCriticTrainer::default_batch_mode = true;
const unsigned int GreedyActorCriticTrainer::default_batch_size = 0;
const int GreedyActorCriticTrainer::default_report_frequency = 1;

const string GreedyActorCriticTrainer::delta_network_weights_id =
      "delta_network_weights";
const string GreedyActorCriticTrainer::previous_delta_network_weights_id =
      "previous_delta_network_weights";
const string GreedyActorCriticTrainer::adjusted_network_weights_id =
      "adjusted_network_weights";
const string GreedyActorCriticTrainer::best_weights_id = "best_weights";
const string GreedyActorCriticTrainer::failback_weights_id = "failback_weights";

GreedyActorCriticTrainer::GreedyActorCriticTrainer(GreedyActorCriticNet& _model) :
      actor_critic_model(_model)
{
   srand(time(NULL));

   set_max_epochs(default_max_epochs);
   set_learning_momentum(default_learning_momentum);
   set_performance_goal(default_performance_goal);
   set_min_gradient(default_min_gradient);
   set_max_validation_failures(default_max_valid_fail);
   set_batch_mode();
   set_report_frequency(default_report_frequency);

   predict_mode = FINAL_COST;

   set_verbose(false);
   set_global_learning_rate(0.001);
   set_critic_batch_size(10);
   set_actor_batch_size(10);

   initialized = false;
}

GreedyActorCriticTrainer::~GreedyActorCriticTrainer()
{
   // TODO Auto-generated destructor stub
}

NetworkWeightsData& GreedyActorCriticTrainer::get_cached_network_weights(
      const string& _id, const BaseNeuralNet& _net)
{
   string key = genkey(_net, _id);

   if (network_weights_cache.find(key) == network_weights_cache.end())
      alloc_network_weights_cache_entry(_id, _net);

   return network_weights_cache.at(key);
}

void GreedyActorCriticTrainer::set_global_learning_rate(double _rate)
{
   alloc_network_learning_rates();
   network_learning_rates_map.at(actor_critic_model.get_actor()->name())->set_global_learning_rate(
         _rate);
   network_learning_rates_map.at(
         actor_critic_model.get_adaptive_critic()->name())->set_global_learning_rate(
         _rate);
}

void GreedyActorCriticTrainer::set_predict_final_cost()
{
   predict_mode = FINAL_COST;
}

void GreedyActorCriticTrainer::set_predict_cumulative_cost()
{
   predict_mode = CUMULATIVE_COST;
}

double GreedyActorCriticTrainer::sim(EnvironmentSimulator* _env,
      unsigned int _sampleCount)
{
   Pattern state;
   Action action;
   ActorCriticOutput ac_out;

   // Flag whether there was external reinforcement available
   bool external_rflag;

   // External reinforcement signal
   double external_rsig;

   double score = 0;

   for (unsigned int i = 0; i < _sampleCount; i++)
   {
      state = _env->reset();
      while (!_env->is_terminal_state(state))
      {
         ac_out = actor_critic_model(state);
         state = _env->next_state(ac_out.action());
      }
      external_rsig = _env->get_reinforcement(external_rflag);

      if (external_rflag)
         score += external_rsig;
   }

   return score / _sampleCount;
}

double GreedyActorCriticTrainer::sim2(EnvironmentSimulator* _env,
      unsigned int _sampleCount)
{
   Pattern state;
   Action action;
   ActorCriticOutput ac_out;

   // Flag whether there was external reinforcement available
   bool external_rflag;

   // External reinforcement signal
   double external_rsig;

   double model_rsig, prev_model_rsig;

   double sse = 0;

   for (unsigned int i = 0; i < _sampleCount; i++)
   {
      state = _env->reset();
      while (!_env->is_terminal_state(state))
      {
         ac_out = actor_critic_model(state);
         model_rsig = ac_out.reinforcement();

         state = _env->next_state(ac_out.action());
      }
      external_rsig = _env->get_reinforcement(external_rflag);

      if (external_rflag)
         sse += (model_rsig - external_rsig) * (model_rsig - external_rsig);
   }

   return sse / _sampleCount;
}

void GreedyActorCriticTrainer::init_train()
{
   cout << "init train" << endl;

   actor_critic_model.clear_error();
   zero_delta_network_weights();

   ActorNet& actornet = *actor_critic_model.get_actor();
   AdaptiveCriticNet& criticnet = *actor_critic_model.get_adaptive_critic();

   network_learning_rates_map.at(actornet.name())->reset();
   network_learning_rates_map.at(criticnet.name())->reset();

   save_weights(failback_weights_id);

   const vector<BaseLayer*>& actor_network_layers =
         actornet.get_network_layers();
   for (unsigned int ndx = 0; ndx < actor_network_layers.size(); ndx++)
   {
      BaseLayer& layer = *actor_network_layers[ndx];
      const string& name = layer.name();
      unsigned int output_sz = layer.size();
      unsigned int input_sz = layer.input_size();

      string key = genkey(actornet, name);

      cout << "init " << key << endl;

      etrace_dAdB_map[key] = Array<double>();
      etrace_dAdB_map[key].resize(output_sz, output_sz);

      cumulative_dAdN_map[key] = Array<double>();
      cumulative_dAdN_map[key].resize(output_sz, output_sz);

      etrace_dAdW_map[key] = Array<double>();
      etrace_dAdW_map[key].resize(output_sz, input_sz);

      cumulative_dNdW_map[key] = Array<double>();
      cumulative_dNdW_map[key].resize(output_sz, input_sz);

      prev_etrace_dAdB_map[key] = Array<double>();
      prev_etrace_dAdB_map[key].resize(output_sz, output_sz);

      prev_cumulative_dAdN_map[key] = Array<double>();
      prev_cumulative_dAdN_map[key].resize(output_sz, output_sz);

      prev_etrace_dAdW_map[key] = Array<double>();
      prev_etrace_dAdW_map[key].resize(output_sz, input_sz);

      prev_cumulative_dNdW_map[key] = Array<double>();
      prev_cumulative_dNdW_map[key].resize(output_sz, input_sz);
   }

   const vector<BaseLayer*>& critic_network_layers =
         criticnet.get_network_layers();

   for (unsigned int ndx = 0; ndx < critic_network_layers.size(); ndx++)
   {
      BaseLayer& layer = *critic_network_layers[ndx];
      const string& name = layer.name();
      unsigned int output_sz = layer.size();
      unsigned int input_sz = layer.input_size();

      string key = genkey(criticnet, name);

      cout << "init " << key << endl;

      etrace_dAdB_map[key] = Array<double>();
      etrace_dAdB_map[key].resize(output_sz, output_sz);

      cumulative_dAdN_map[key] = Array<double>();
      cumulative_dAdN_map[key].resize(output_sz, output_sz);

      etrace_dAdW_map[key] = Array<double>();
      etrace_dAdW_map[key].resize(output_sz, input_sz);

      cumulative_dNdW_map[key] = Array<double>();
      cumulative_dNdW_map[key].resize(output_sz, input_sz);

      prev_etrace_dAdB_map[key] = Array<double>();
      prev_etrace_dAdB_map[key].resize(output_sz, output_sz);

      prev_cumulative_dAdN_map[key] = Array<double>();
      prev_cumulative_dAdN_map[key].resize(output_sz, output_sz);

      prev_etrace_dAdW_map[key] = Array<double>();
      prev_etrace_dAdW_map[key].resize(output_sz, input_sz);

      prev_cumulative_dNdW_map[key] = Array<double>();
      prev_cumulative_dNdW_map[key].resize(output_sz, input_sz);
   }
}

void GreedyActorCriticTrainer::train(EnvironmentSimulator* _trainingEnv,
      double _objVal)
{
   double perf, prev_perf;
   double global_performance = 0.5;
   double best_global_performance = 0;
   double score, mse, prev_mse = 10000;
   double best_score = 0, best_mse = 10000.0;

   bool update_actor_flag = false;

   int consecutive_failback_count = 0;

   init_train();

   for (unsigned int epoch = 0; epoch < max_training_epochs; epoch++)
   {
      init_training_epoch();

      update_actor_flag =
            (epoch % 1 == 0 || epoch == max_training_epochs - 1) ? true : false;

      // train one sequence/game
      //perf = train_exemplar(_trainingEnv, _objVal, update_actor_flag);
      perf = train_exemplar(_trainingEnv, _objVal, false);

      double alpha = 0.25;
      global_performance = alpha * perf + (1 - alpha) * global_performance;

      //if (is_online_mode())
      {
         apply_learning_rate_adjustments();
         //apply_delta_network_weights();

         apply_delta_network_weights(*actor_critic_model.get_adaptive_critic());
         zero_delta_network_weights(*actor_critic_model.get_adaptive_critic());

         if (update_actor_flag)
         {
            apply_delta_network_weights(*actor_critic_model.get_actor());
            zero_delta_network_weights(*actor_critic_model.get_actor());
         }

         //zero_delta_network_weights();
         zero_network_eligibility_trace();
      }

      score = sim(_trainingEnv, 500);
      mse = sim2(_trainingEnv, 500);

      if (is_verbose_mode() && epoch > 0)
         cout << "global perf(" << epoch << ") = " << global_performance
               << "; score = " << score << "; r mse = " << mse << endl;

      if ((mse - prev_mse) / prev_mse > 0.2)
      {
         cout << "fail back weights because of large increase in error. "
               << prev_mse << " => " << mse << endl;

         /*
         BaseNeuralNet* cnet = actor_critic_model.get_adaptive_critic();

         const vector<BaseLayer*> cnet_layers = cnet->get_network_layers();
         for (unsigned int ndx = 0; ndx < cnet_layers.size(); ndx++)
         {
            BaseLayer& layer = *cnet_layers[ndx];
            const string& lname = layer.name();

            Array<double> layer_weights_lr =
                  network_learning_rates_map.at(cnet->name())->get_weight_learning_rates().at(
                        lname);

            printf("learning rates (%s)\n", lname.c_str());
            for (unsigned int i=0; i<layer_weights_lr.rowDim(); i++)
            {
               for (unsigned int j=0; j<layer_weights_lr.colDim(); j++)
               {
                  printf("%f ", layer_weights_lr[i][j]);
               }
               printf("\n");
            }
            printf("------------------------------\n\n");
         }
         */

         consecutive_failback_count++;
         if (consecutive_failback_count > 25)
         {
            cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
            break;
         }

         restore_weights(failback_weights_id);
         reduce_learning_rates(0.4);

         continue;
      }
      else
      {
         prev_mse = mse;
         consecutive_failback_count = 0;
      }

      /*
       if (score > best_score)
       {
       best_score = score;
       save_weights(best_weights_id);
       }
       */

      if (mse < best_mse)
      {
         best_mse = mse;
         save_weights(best_weights_id);
      }
   }

   // Restore weights to the best training epoch
   restore_weights(best_weights_id);

   score = sim(_trainingEnv, 500);
   mse = sim2(_trainingEnv, 500);

   if (is_verbose_mode())
      cout << "best global perf = " << global_performance << "; score = "
            << score << "; r mse = " << mse << endl;

}

/*
 void GreedyActorCriticTrainer::train(EnvironmentSimulator* _trainingEnv,
 double _objVal)
 {
 double perf;
 double global_performance = 0.5;
 double best_global_performance = 0;
 double score;
 double best_score = 0;

 bool update_actor_flag = false;

 double alpha = 0.01;

 init_train();

 save_weights(best_weights_id);

 for (unsigned int epoch = 0; epoch < max_training_epochs; epoch++)
 {
 init_training_epoch();

 train_critic(_trainingEnv, critic_batch_size);
 perf = train_actor(_trainingEnv, _objVal, actor_batch_size);

 global_performance = alpha * perf + (1 - alpha) * global_performance;

 score = sim(_trainingEnv, 501);

 if (is_verbose_mode() && epoch > 0)
 cout << "global perf(" << epoch << ") = " << global_performance
 << "; score = " << score << endl;

 if (score > best_score)
 {
 best_score = score;
 save_weights(best_weights_id);
 }
 }

 // Restore weights to the best training epoch
 restore_weights(best_weights_id);
 }
 */

double GreedyActorCriticTrainer::train_critic(
      EnvironmentSimulator* _trainingEnv, unsigned int _batchSize)
{
   double perf;

   zero_delta_network_weights(*actor_critic_model.get_adaptive_critic());
   for (unsigned int epoch = 0; epoch < _batchSize; epoch++)
   {
      actor_critic_model.clear_error();
      zero_network_eligibility_trace();

      perf = train_exemplar(_trainingEnv, 1, false);
   }

   apply_delta_network_weights(*actor_critic_model.get_adaptive_critic());
   zero_delta_network_weights(*actor_critic_model.get_adaptive_critic());

   return 0;
}

double GreedyActorCriticTrainer::train_actor(EnvironmentSimulator* _trainingEnv,
      double _objVal, unsigned int _batchSize)
{
   double perf = 0;

   zero_delta_network_weights(*actor_critic_model.get_actor());
   for (unsigned int epoch = 0; epoch < _batchSize; epoch++)
   {
      actor_critic_model.clear_error();
      zero_network_eligibility_trace();

      perf += train_exemplar(_trainingEnv, _objVal, true);
   }

   apply_delta_network_weights(*actor_critic_model.get_actor());
   zero_delta_network_weights(*actor_critic_model.get_actor());

   return perf / _batchSize;
}

void GreedyActorCriticTrainer::init_training_epoch()
{
   actor_critic_model.clear_error();
   zero_delta_network_weights();
   zero_network_eligibility_trace();
   save_weights(failback_weights_id);
}

/**
 * Train one sequence/game until a terminal state is reached
 */
double GreedyActorCriticTrainer::train_exemplar(EnvironmentSimulator * _env,
      double _objVal, bool _updateActorFlag)
{
   unsigned int step_no = 0;
   Pattern state, prev_state, next_state;
   ActorCriticOutput ac_out, prev_ac_out;

   // Flag whether there was external reinforcement available
   bool external_rflag;

   // External reinforcement signal
   double external_rsig;

   // Reinforcement signal to use for training
   double training_rsig;

   // Performance variables
   double patt_sse;
   double patt_err, prev_patt_err = 0;
   double seq_sse = 0;

   vector<double> ugradient(1, 1.0);

   vector<double> egradient(1);
   vector<double> prev_opatt(1);
   vector<double> tgt_patt(1);

   /******************************************************************
    * Set up model, state, and training data on initial state for
    * training on subsequent states
    *
    */
   // Reset the environment to start a new "game"
   state = _env->reset();

   // Initial activation of the model with the start state
   ac_out = actor_critic_model(state);

   /*
    * Temporal difference learning needs to save the old network
    * weight gradients and add the new weight gradients to the
    * discounted sum of the weight gradients
    */
   save_network_eligibility_trace();

   actor_critic_model.clear_error();
   actor_critic_model.backprop(ugradient);
   update_network_eligibility_trace();

   step_no++;

   /***************************************************
    * Train on subsequent states
    */
   do
   {
      // Save current input
      prev_state = state;

      // Update the environment with the recommended action
      state = _env->next_state(ac_out.action());

      // Check for external reinforcement signal
      external_rsig = _env->get_reinforcement(external_rflag);

      // Activate the model with the current state vector
      prev_ac_out = ac_out;

      if (!_env->is_terminal_state(state))
      {
         ac_out = actor_critic_model(state);

         /*
          * Use the external or internal reinforcement signal to calculate
          * the error gradient we will backprop through the actor-critic
          * network in order to train the critic
          */
         training_rsig =
               (external_rflag) ? external_rsig : ac_out.reinforcement();
      }
      else
         training_rsig = external_rsig;

      /*
       * Temporal difference learning needs to save the old network
       * weight gradients and add the new weight gradients to the
       * discounted sum of the weight gradients
       */
      save_network_eligibility_trace();

      actor_critic_model.clear_error();
      actor_critic_model.backprop(ugradient);
      update_network_eligibility_trace();


      // Adaptive critic can only learn after the first step
      if (step_no > 0)
      {
         // TODO - activate network for previous input state
         //prev_ac_out = actor_critic_model(prev_state);
         prev_opatt[0] = prev_ac_out.reinforcement();

         if (predict_mode == FINAL_COST)
         {
            tgt_patt[0] = training_rsig;
            //cout << "final_cost training sig " << tgt_patt.at(0) << endl;
            error_func(patt_sse, egradient, prev_opatt, tgt_patt);
            //cout << "egradient " << egradient[0] << endl;
         }
         else if (predict_mode == CUMULATIVE_COST)
         {

            if (_env->is_terminal_state(state))
            {
               patt_err = -(external_rsig - prev_opatt.at(0));
            }
            else
            {
               patt_err = -(0.95 * ac_out.reinforcement() + external_rsig
                     - prev_opatt.at(0));
            }

            //egradient[0] = patt_err - 0.95 * prev_patt_err; wtf?? 7-19-2015
            egradient[0] = patt_err;

            patt_sse = 0.5 * (patt_err * patt_err);
         }
         //seq_sse += patt_sse;

         // TODO - Should I update the actor and critic rates separately?
         network_learning_rates_map.at(
               actor_critic_model.get_adaptive_critic()->name())->update_learning_rate_adjustments();

         calc_network_adj(*actor_critic_model.get_adaptive_critic(), egradient);
      }

      //if (_updateActorFlag && _env->is_terminal_state(state))
      if (_updateActorFlag)
      {
         /*
          * Use the ultimate objective value to calculate the error gradient
          * we will backprop through the actor-critic network in order to train
          * the actor network
          */
         tgt_patt[0] = _objVal;
         error_func(patt_sse, egradient, prev_opatt, tgt_patt);

         network_learning_rates_map.at(actor_critic_model.get_actor()->name())->update_learning_rate_adjustments();

         calc_network_adj(*actor_critic_model.get_actor(), egradient);
      }

      step_no++;

   } while (!_env->is_terminal_state(state));
   //seq_sse = (step_no > 0) ? seq_sse / step_no : 0;
   seq_sse = external_rsig;

   return seq_sse;
}

void GreedyActorCriticTrainer::calc_network_adj(const BaseNeuralNet& _net,
      const vector<double>& errorv)
{
   NetworkWeightsData& network_deltas = get_cached_network_weights(
         delta_network_weights_id, _net);

   const vector<BaseLayer*> network_layers = _net.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      const string& name = layer.name();

      LayerWeightsData& layer_deltas = network_deltas.layer_weights(name);

      calc_layer_bias_adj(_net, layer, 1, layer_deltas.biases, errorv);
      calc_layer_weight_adj(_net, layer, 1, layer_deltas.weights, errorv);
   }
}

void GreedyActorCriticTrainer::calc_layer_bias_adj(const BaseNeuralNet& _net,
      const BaseLayer& layer, unsigned int timeStep, vector<double>& biasDelta,
      const vector<double>& errorv)
{
   string key = genkey(_net, layer.name());

   Array<double>& etrace_dAdB = prev_etrace_dAdB_map[key];

   // Get the learning rates for the biases
   vector<double> layer_bias_lr =
         network_learning_rates_map.at(_net.name())->get_bias_learning_rates().at(
               layer.name());

   for (unsigned int bias_ndx = 0; bias_ndx < biasDelta.size(); bias_ndx++)
      for (unsigned int err_ndx = 0; err_ndx < errorv.size(); err_ndx++)
         biasDelta.at(bias_ndx) += -layer_bias_lr[bias_ndx] * errorv.at(err_ndx)
               * etrace_dAdB.at(bias_ndx, bias_ndx);
}

void GreedyActorCriticTrainer::calc_layer_weight_adj(const BaseNeuralNet& _net,
      const BaseLayer& layer, unsigned int timeStep, Array<double>& weightDelta,
      const vector<double>& errorv)
{
   string key = genkey(_net, layer.name());

   // Get the learning rates for the weights
   Array<double> layer_weights_lr =
         network_learning_rates_map.at(_net.name())->get_weight_learning_rates().at(
               layer.name());

   unsigned int layer_size = layer.size();
   unsigned int layer_input_size = layer.input_size();

   Array<double>& etrace_dAdW = prev_etrace_dAdW_map[key];
   for (unsigned int out_ndx = 0; out_ndx < layer.size(); out_ndx++)
   {
      for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
      {
         for (unsigned int err_ndx = 0; err_ndx < errorv.size(); err_ndx++)
            weightDelta.at(out_ndx, in_ndx) += -layer_weights_lr.at(out_ndx,
                  in_ndx) * errorv.at(err_ndx)
                  * etrace_dAdW.at(out_ndx, in_ndx);
      }
   }

   /*
    Array<double>& cumulative_dAdN = prev_cumulative_dAdN_map[key];
    Array<double>& cumulative_dNdW = prev_cumulative_dNdW_map[key];
    double temp;
    for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
    {
    temp = 0;
    for (unsigned int out_ndx = 0; out_ndx < layer.size(); out_ndx++)
    temp += errorv.at(out_ndx) * cumulative_dAdN.at(out_ndx, netin_ndx);

    for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
    weightDelta.at(netin_ndx, in_ndx) += layer_weights_lr.at(netin_ndx,
    in_ndx) * temp * cumulative_dNdW.at(netin_ndx, in_ndx);
    }
    */
}

void GreedyActorCriticTrainer::save_weights(const string& _id)
{
   save_weights(_id, *actor_critic_model.get_actor());
   save_weights(_id, *actor_critic_model.get_adaptive_critic());
}

void GreedyActorCriticTrainer::save_weights(const string& _id,
      const BaseNeuralNet& _net)
{
   // TODO - check the buffer id to make sure it's not one of the reserved values
   NetworkWeightsData& network_weights_data = get_cached_network_weights(_id,
         _net);

   const vector<BaseLayer*> network_layers = _net.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      LayerWeightsData& layer_weights_data = network_weights_data.layer_weights(
            name);

      // Layer value at time step 0 is the initial layer value
      layer_weights_data.initial_value = layer(0);

      layer_weights_data.biases = layer.get_biases();
      layer_weights_data.weights = layer.get_weights();
   }
}

void GreedyActorCriticTrainer::restore_weights(const string& _id)
{
   restore_weights(_id, *actor_critic_model.get_actor());
   restore_weights(_id, *actor_critic_model.get_adaptive_critic());
}

void GreedyActorCriticTrainer::restore_weights(const string& _id,
      const BaseNeuralNet& _net)
{
   // TODO - check the buffer id to make sure it's not one of the reserved values
   string key = genkey(_net, _id);
   NetworkWeightsData& network_weights_data = network_weights_cache[key];

   const vector<BaseLayer*> network_layers = _net.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      LayerWeightsData& layer_weights_data = network_weights_data.layer_weights(
            name);

      layer.set_biases(layer_weights_data.biases);
      layer.set_weights(layer_weights_data.weights);
   }
}

void GreedyActorCriticTrainer::alloc_network_weights_cache_entry(
      const string& _id)
{
   alloc_network_weights_cache_entry(_id, *actor_critic_model.get_actor());
   alloc_network_weights_cache_entry(_id,
         *actor_critic_model.get_adaptive_critic());
}

void GreedyActorCriticTrainer::alloc_network_weights_cache_entry(
      const string& _id, const BaseNeuralNet& _net)
{
   string key = genkey(_net, _id);
   NetworkWeightsData& network_weights = network_weights_cache[key];

   const vector<BaseLayer*> network_layers = _net.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      const string& name = layer.name();

      const Array<double>& layer_weights = layer.get_weights();
      unsigned int rows = layer_weights.rowDim();
      unsigned int cols = layer_weights.colDim();

      LayerWeightsData& layer_weights_data = network_weights.new_layer_weights(
            name);

      layer_weights_data.initial_value.resize(layer.size(), 0.0);
      layer_weights_data.biases.resize(layer.get_biases().size(), 0.0);

      layer_weights_data.weights.resize(rows, cols);
      layer_weights_data.weights = 0;
   }
}

void GreedyActorCriticTrainer::apply_delta_network_weights()
{
   apply_delta_network_weights(*actor_critic_model.get_actor());
   apply_delta_network_weights(*actor_critic_model.get_adaptive_critic());
}

void GreedyActorCriticTrainer::apply_delta_network_weights(BaseNeuralNet& _net)
{
   NetworkWeightsData& delta_network_weights = get_cached_network_weights(
         delta_network_weights_id, _net);
   NetworkWeightsData& prev_delta_network_weights = get_cached_network_weights(
         previous_delta_network_weights_id, _net);
   NetworkWeightsData& adjusted_network_weights = get_cached_network_weights(
         adjusted_network_weights_id, _net);

   const vector<BaseLayer*> network_layers = _net.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      const LayerWeightsData& delta_layer_weights_data =
            delta_network_weights.layer_weights(name);
      const LayerWeightsData& prev_delta_layer_weights_data =
            prev_delta_network_weights.layer_weights(name);
      LayerWeightsData& adjusted_layer_weights_data =
            adjusted_network_weights.layer_weights(name);

      const vector<double>& delta_layer_biases = delta_layer_weights_data.biases;
      const vector<double>& prev_delta_layer_biases =
            prev_delta_layer_weights_data.biases;
      vector<double>& adjusted_layer_biases = adjusted_layer_weights_data.biases;

      adjusted_layer_biases = layer.get_biases();
      for (unsigned int ndx = 0; ndx < adjusted_layer_biases.size(); ndx++)
         adjusted_layer_biases.at(ndx) += learning_momentum
               * prev_delta_layer_biases.at(ndx) + delta_layer_biases.at(ndx);

      if (layer.is_learn_biases())
         layer.set_biases(adjusted_layer_biases);

      const Array<double>& delta_layer_weights =
            delta_layer_weights_data.weights;
      const Array<double>& prev_delta_layer_weights =
            prev_delta_layer_weights_data.weights;
      Array<double>& adjusted_layer_weights =
            adjusted_layer_weights_data.weights;

      adjusted_layer_weights = layer.get_weights();

      unsigned int row_sz = adjusted_layer_weights.rowDim();
      unsigned int col_sz = adjusted_layer_weights.colDim();

      for (unsigned int row = 0; row < row_sz; row++)
      {
         for (unsigned int col = 0; col < col_sz; col++)
         {
            adjusted_layer_weights.at(row, col) += learning_momentum
                  * prev_delta_layer_weights.at(row, col)
                  + delta_layer_weights.at(row, col);
         }
      }

      if (layer.is_learn_weights())
         layer.set_weights(adjusted_layer_weights);
   }

   // TODO - decide if we should save off the current delta weights here
   prev_delta_network_weights = delta_network_weights;
}

void GreedyActorCriticTrainer::zero_delta_network_weights()
{
   zero_delta_network_weights(*actor_critic_model.get_actor());
   zero_delta_network_weights(*actor_critic_model.get_adaptive_critic());
}

void GreedyActorCriticTrainer::zero_delta_network_weights(
      const BaseNeuralNet& _net)
{
   // Get a reference from the buffer to the entry containing the delta
   // network weights
   NetworkWeightsData& delta_network_weights = get_cached_network_weights(
         delta_network_weights_id, _net);

   // Iterate through the weights for each layer

   const set<string>& key_set = delta_network_weights.keySet();
   set<string>::iterator iter;
   for (iter = key_set.begin(); iter != key_set.end(); iter++)
   {
      const string& key_str = *iter;
      LayerWeightsData& layer_weights_data =
            delta_network_weights.layer_weights(key_str);

      unsigned int sz;

      // Clear deltas for initial layer value
      sz = layer_weights_data.initial_value.size();
      for (unsigned int ndx = 0; ndx < sz; ndx++)
         layer_weights_data.initial_value[ndx] = 0;

      // Clear deltas for layer biases
      sz = layer_weights_data.biases.size();
      for (unsigned int ndx = 0; ndx < sz; ndx++)
         layer_weights_data.biases[ndx] = 0;

      // Clear deltas for layer weights
      layer_weights_data.weights = 0;
   }
}

int GreedyActorCriticTrainer::urand(int n)
{
   int top = ((((RAND_MAX - n) + 1) / n) * n - 1) + n;
   int r;
   do
   {
      r = rand();
   } while (r > top);
   return (r % n);
}

inline
void GreedyActorCriticTrainer::save_network_eligibility_trace()
{
   save_network_eligibility_trace(*actor_critic_model.get_actor());
   save_network_eligibility_trace(*actor_critic_model.get_adaptive_critic());
}

inline
void GreedyActorCriticTrainer::save_network_eligibility_trace(
      const BaseNeuralNet& _net)
{
   string key;

   const vector<BaseLayer*> network_layers = _net.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      key = genkey(_net, name);

      // Save etrace dAdB
      Array<double>& etrace_dAdB = etrace_dAdB_map[key];
      Array<double>& prev_etrace_dAdB = prev_etrace_dAdB_map[key];

      prev_etrace_dAdB = etrace_dAdB;

      // Save etrace dAdW
      Array<double>& etrace_dAdW = etrace_dAdW_map[key];
      Array<double>& prev_etrace_dAdW = prev_etrace_dAdW_map[key];

      prev_etrace_dAdW = etrace_dAdW;

      // Save cumulative dAdN
      Array<double>& cumulative_dAdN = cumulative_dAdN_map[key];
      Array<double>& prev_cumulative_dAdN = prev_cumulative_dAdN_map[key];

      prev_cumulative_dAdN = cumulative_dAdN;

      // Save cumulative dNdW
      Array<double>& cumulative_dNdW = cumulative_dNdW_map[key];
      Array<double>& prev_cumulative_dNdW = prev_cumulative_dNdW_map[key];

      prev_cumulative_dNdW = cumulative_dNdW;
   }
}

inline
void GreedyActorCriticTrainer::update_network_eligibility_trace()
{
   update_network_eligibility_trace(*actor_critic_model.get_actor());
   update_network_eligibility_trace(*actor_critic_model.get_adaptive_critic());
}

inline
void GreedyActorCriticTrainer::update_network_eligibility_trace(
      const BaseNeuralNet& _net)
{
   string key;

   const vector<BaseLayer*> network_layers = _net.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      key = genkey(_net, name);

      double prev_sum;
      double temp = 0;

      Array<double>& etrace_dAdB = etrace_dAdB_map[key];
      const vector<double>& dEdB = layer.get_dEdB();

      for (unsigned int bias_ndx = 0; bias_ndx < dEdB.size(); bias_ndx++)
      {
         prev_sum = etrace_dAdB.at(bias_ndx, bias_ndx);
         etrace_dAdB.at(bias_ndx, bias_ndx) = dEdB.at(bias_ndx)
               + lambda * prev_sum;
      }

      Array<double>& etrace_dAdW = etrace_dAdW_map[key];
      const Array<double>& dEdW = layer.get_dEdW();
      const Array<double>& dNdW = layer.get_dNdW();
      const Array<double>& dAdN = layer.get_dAdN();

      unsigned int layer_size = layer.size();
      unsigned int layer_input_size = layer.input_size();

      for (unsigned int out_ndx = 0; out_ndx < layer.size(); out_ndx++)
      {
         for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
         {
            prev_sum = etrace_dAdW.at(out_ndx, in_ndx);
            etrace_dAdW.at(out_ndx, in_ndx) = dEdW.at(out_ndx, in_ndx)
                  + lambda * prev_sum;
         }
      }

      /*
       for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
       {
       temp = 0;
       for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
       temp += dAdN.at(out_ndx, netin_ndx);

       for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
       {
       prev_sum = etrace_dAdW.at(netin_ndx, in_ndx);
       etrace_dAdW.at(netin_ndx, in_ndx) = temp
       * dNdW.at(netin_ndx, in_ndx) + lambda * prev_sum;
       }
       }
       */

      /*
       Array<double>& cumulative_dAdN = cumulative_dAdN_map[key];
       for (unsigned int out_ndx = 0; out_ndx < layer_size; out_ndx++)
       {
       for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
       {
       prev_sum = cumulative_dAdN.at(out_ndx, netin_ndx);
       cumulative_dAdN.at(out_ndx, netin_ndx) = dAdN.at(out_ndx, netin_ndx)
       + lambda * prev_sum;
       }
       }

       Array<double>& cumulative_dNdW = cumulative_dNdW_map[key];
       for (unsigned int netin_ndx = 0; netin_ndx < layer_size; netin_ndx++)
       {
       for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
       {
       prev_sum = cumulative_dNdW.at(netin_ndx, in_ndx);
       cumulative_dNdW.at(netin_ndx, in_ndx) = dNdW.at(netin_ndx, in_ndx)
       + lambda * prev_sum;
       }
       }
       */
   }
}

void GreedyActorCriticTrainer::zero_network_eligibility_trace()
{
   zero_network_eligibility_trace(*actor_critic_model.get_actor());
   zero_network_eligibility_trace(*actor_critic_model.get_adaptive_critic());
}

void GreedyActorCriticTrainer::zero_network_eligibility_trace(
      const BaseNeuralNet& _net)
{
   /* ******************************************
    *    Zero bias and weight deltas
    */
   const vector<BaseLayer*> network_layers = _net.get_network_layers();
   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      const string& name = layer.name();

      Array<double>& etrace_dAdB = etrace_dAdB_map[name];
      Array<double>& etrace_dAdW = etrace_dAdW_map[name];

      Array<double>& cumulative_dAdN = cumulative_dAdN_map[name];
      Array<double>& cumulative_dNdW = cumulative_dNdW_map[name];

      etrace_dAdB = 0;
      etrace_dAdW = 0;

      cumulative_dAdN = 0;
      cumulative_dNdW = 0;
   }
}

void GreedyActorCriticTrainer::alloc_network_learning_rates()
{
   const BaseNeuralNet& actor = *actor_critic_model.get_actor();
   const BaseNeuralNet& critic = *actor_critic_model.get_adaptive_critic();

   if (network_learning_rates_map.find(actor.name())
         == network_learning_rates_map.end())
      //network_learning_rates_map[actor.name()] = new ConstantLearningRate(
      //      actor);
      network_learning_rates_map[actor.name()] = new DeltaBarDeltaLearningRate(
            actor);

   if (network_learning_rates_map.find(critic.name())
         == network_learning_rates_map.end())
      //network_learning_rates_map[critic.name()] = new ConstantLearningRate(
      //      critic);
      network_learning_rates_map[critic.name()] = new DeltaBarDeltaLearningRate(
            critic);
}

void GreedyActorCriticTrainer::apply_learning_rate_adjustments()
{
   const BaseNeuralNet& actor = *actor_critic_model.get_actor();
   const BaseNeuralNet& critic = *actor_critic_model.get_adaptive_critic();

   if (network_learning_rates_map.find(actor.name())
         != network_learning_rates_map.end())
      network_learning_rates_map[actor.name()]->apply_learning_rate_adjustments();

   if (network_learning_rates_map.find(critic.name())
         != network_learning_rates_map.end())
      network_learning_rates_map[critic.name()]->apply_learning_rate_adjustments();

}

void GreedyActorCriticTrainer::reduce_learning_rates(double val)
{
   const BaseNeuralNet& actor = *actor_critic_model.get_actor();
   const BaseNeuralNet& critic = *actor_critic_model.get_adaptive_critic();

   if (network_learning_rates_map.find(actor.name())
         != network_learning_rates_map.end())
      network_learning_rates_map[actor.name()]->reduce_learning_rate(val);

   if (network_learning_rates_map.find(critic.name())
         != network_learning_rates_map.end())
      network_learning_rates_map[critic.name()]->reduce_learning_rate(val);

}

} /* namespace flex_neuralnet */
