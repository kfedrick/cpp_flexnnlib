/*
 * ActorCriticTrainer3.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#include "ActorCriticTrainer3.h"

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace std;

namespace flexnnet
{

   const long ActorCriticTrainer3::default_max_epochs = 1;
   const double ActorCriticTrainer3::default_learning_momentum = 0.0;
   const double ActorCriticTrainer3::default_performance_goal = 0.0;
   const double ActorCriticTrainer3::default_min_gradient = 0.0;
   const int ActorCriticTrainer3::default_max_valid_fail = 10;
   const bool ActorCriticTrainer3::default_batch_mode = true;
   const unsigned int ActorCriticTrainer3::default_batch_size = 0;
   const int ActorCriticTrainer3::default_report_frequency = 1;

   const string ActorCriticTrainer3::delta_network_weights_id =
      "delta_network_weights";
   const string ActorCriticTrainer3::previous_delta_network_weights_id =
      "previous_delta_network_weights";
   const string ActorCriticTrainer3::adjusted_network_weights_id =
      "adjusted_network_weights";
   const string ActorCriticTrainer3::best_weights_id = "best_weights";
   const string ActorCriticTrainer3::failback_weights_id = "failback_weights";

   ActorCriticTrainer3::ActorCriticTrainer3 (ActorCriticNet3 &_model) :
      actor_critic_model (_model)
   {
      srand (time (NULL));

      set_max_epochs (default_max_epochs);
      set_learning_momentum (default_learning_momentum);
      set_performance_goal (default_performance_goal);
      set_min_gradient (default_min_gradient);
      set_max_validation_failures (default_max_valid_fail);
      set_batch_mode ();
      set_report_frequency (default_report_frequency);

      predict_mode = CUMULATIVE_COST;

      set_verbose (false);
      set_global_learning_rate (0.001);
      set_critic_batch_size (10);
      set_actor_batch_size (10);

      set_print_gradient (false);

      initialized = false;
   }

   ActorCriticTrainer3::~ActorCriticTrainer3 ()
   {
      // TODO Auto-generated destructor stub
   }

   NetworkWeightsData &ActorCriticTrainer3::get_cached_network_weights (
      const string &_id, const TDCNeuralNet &_net)
   {
      string key = genkey (_net, _id);

      if (network_weights_cache.find (key) == network_weights_cache.end ())
         alloc_network_weights_cache_entry (_id, _net);

      return network_weights_cache.at (key);
   }

   void ActorCriticTrainer3::set_global_learning_rate (double _rate)
   {
      alloc_network_learning_rates ();
      network_learning_rates_map.at (actor_critic_model.get_actor ()->name ())->set_global_learning_rate (
         _rate);
      network_learning_rates_map.at (
         actor_critic_model.get_adaptive_critic ()->name ())->set_global_learning_rate (
         _rate);
   }

   void ActorCriticTrainer3::set_actor_learning_rate (double _rate)
   {
      network_learning_rates_map.at (actor_critic_model.get_actor ()->name ())->set_global_learning_rate (
         _rate);
   }

   void ActorCriticTrainer3::set_critic_learning_rate (double _rate)
   {
      network_learning_rates_map.at (
         actor_critic_model.get_adaptive_critic ()->name ())->set_global_learning_rate (
         _rate);
   }

   double ActorCriticTrainer3::sim (EnvironmentSimulator *_env,
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
         state = _env->reset ();
         while (!_env->is_terminal_state (state))
         {
            ac_out = actor_critic_model (state);
            state = _env->next_state (ac_out.action ());
         }
         external_rsig = _env->get_reinforcement (external_rflag);

         if (external_rflag)
            score += external_rsig;
      }

      return score / _sampleCount;
   }

   double ActorCriticTrainer3::sim2 (EnvironmentSimulator *_env,
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
         state = _env->reset ();
         while (!_env->is_terminal_state (state))
         {
            ac_out = actor_critic_model (state, _env->hint ());
            state = _env->next_state (ac_out.action ());
         }
         external_rsig = _env->get_reinforcement (external_rflag);

         if (external_rflag)
            score += external_rsig;
      }

      return score / _sampleCount;
   }

   double ActorCriticTrainer3::sim_critic (EnvironmentSimulator *_env,
                                           unsigned int _sampleCount)
   {
      unsigned int step = 0;
      Pattern state;
      Action action;
      ActorCriticOutput ac_out, prev_ac_out;

      // Flag whether there was external reinforcement available
      bool external_rflag;

      // External reinforcement signal
      double external_rsig;

      double td_error;
      double patt_sse;
      double sse = 0, mse = 0, trial_mse = 0;
      double training_rsig;

      vector<double> ugradient (1, 1.0);

      vector<double> egradient (1);
      vector<double> prev_opatt (1);
      vector<double> tgt_patt (1);

      for (unsigned int i = 0; i < _sampleCount; i++)
      {
         trial_mse = 0;

         state = _env->reset ();
         ac_out = actor_critic_model (state);

         step = 1;
         do
         {

            /*
             * Update state and get external reinforcement, if any
             */
            state = _env->next_state (ac_out.action ());

            /*
             * Save previous output and reinforcement
             */
            prev_ac_out = ac_out;
            prev_opatt[0] = prev_ac_out.reinforcement ();

            /*
             * Get next action and predicted reinforcement
             */
            ac_out = actor_critic_model (state);

            if (step > 0)
            {
               external_rsig = _env->get_reinforcement (external_rflag);

               if (!_env->is_terminal_state (state))
                  training_rsig =
                     (external_rflag) ? external_rsig : ac_out.reinforcement ();
               else
                  training_rsig = external_rsig;

               if (predict_mode == FINAL_COST)
               {
                  tgt_patt[0] = training_rsig;
                  error_func (patt_sse, egradient, prev_opatt, tgt_patt);

                  if (print_gradient)
                  {
                     cout << "tgt patt = " << tgt_patt[0] << endl;
                     cout << "prev opatt = " << prev_opatt[0] << endl;
                  }
               }
               else if (predict_mode == CUMULATIVE_COST)
               {

                  if (_env->is_terminal_state (state))
                     td_error = external_rsig - prev_opatt.at (0);
                  else
                     td_error = external_rsig + gamma * ac_out.reinforcement ()
                                - prev_opatt.at (0);

                  egradient[0] = td_error;
                  patt_sse = 0.5 * (td_error * td_error);
               }

               trial_mse += patt_sse;
            }

            step++;

         }
         while (!_env->is_terminal_state (state));

         trial_mse = trial_mse / (step - 1);
         mse += trial_mse;
      }

      mse = mse / _sampleCount;
      return mse;
   }

   double ActorCriticTrainer3::sim_critic2 (EnvironmentSimulator *_env,
                                            unsigned int _sampleCount)
   {
      unsigned int step = 0;
      Pattern state;
      Action action;
      ActorCriticOutput ac_out, prev_ac_out;

      // Flag whether there was external reinforcement available
      bool external_rflag;

      // External reinforcement signal
      double external_rsig;

      double td_error;
      double patt_sse;
      double sse = 0, mse = 0, trial_mse = 0;
      double training_rsig;

      vector<double> ugradient (1, 1.0);

      vector<double> egradient (1);
      vector<double> prev_opatt (1);
      vector<double> tgt_patt (1);

      for (unsigned int i = 0; i < _sampleCount; i++)
      {
         trial_mse = 0;

         state = _env->reset ();
         ac_out = actor_critic_model (state, _env->hint ());

         step = 1;
         do
         {

            /*
             * Update state and get external reinforcement, if any
             */
            state = _env->next_state (ac_out.action ());

            /*
             * Save previous output and reinforcement
             */
            prev_ac_out = ac_out;
            prev_opatt[0] = prev_ac_out.reinforcement ();

            /*
             * Get next action and predicted reinforcement
             */
            ac_out = actor_critic_model (state, _env->hint ());

            if (step > 0)
            {
               external_rsig = _env->get_reinforcement (external_rflag);

               if (!_env->is_terminal_state (state))
                  training_rsig =
                     (external_rflag) ? external_rsig : ac_out.reinforcement ();
               else
                  training_rsig = external_rsig;

               if (predict_mode == FINAL_COST)
               {
                  tgt_patt[0] = training_rsig;
                  error_func (patt_sse, egradient, prev_opatt, tgt_patt);

                  if (print_gradient)
                  {
                     cout << "tgt patt = " << tgt_patt[0] << endl;
                     cout << "prev opatt = " << prev_opatt[0] << endl;
                  }
               }
               else if (predict_mode == CUMULATIVE_COST)
               {

                  if (_env->is_terminal_state (state))
                     td_error = external_rsig - prev_opatt.at (0);
                  else
                     td_error = external_rsig + gamma * ac_out.reinforcement ()
                                - prev_opatt.at (0);

                  egradient[0] = td_error;
                  patt_sse = 0.5 * (td_error * td_error);
               }

               trial_mse += patt_sse;
            }

            step++;

         }
         while (!_env->is_terminal_state (state));

         trial_mse = trial_mse / (step - 1);
         mse += trial_mse;
      }

      mse = mse / _sampleCount;
      return mse;
   }

   void ActorCriticTrainer3::save_critic_gradient ()
   {
      TDCNeuralNet &_net = *actor_critic_model.get_adaptive_critic ();

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         prev_critic_dEdB_map[name] = layer.get_dEdB ();
         prev_critic_dEdW_map[name] = layer.get_dEdW ();
         prev_critic_Hv_map[name] = _net.get_Hv (name);
      }
   }

   void ActorCriticTrainer3::save_actor_gradient ()
   {
      TDCNeuralNet &_net = *actor_critic_model.get_actor ();

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         prev_actor_dEdB_map[name] = layer.get_dEdB ();
         prev_actor_dEdW_map[name] = layer.get_dEdW ();
         prev_actor_Hv_map[name] = _net.get_Hv (name);
      }
   }

   void ActorCriticTrainer3::init_train ()
   {
      cout << "init train" << endl;

      actor_critic_model.clear_error ();
      zero_delta_network_weights ();

      ActorNet3 &actornet = *actor_critic_model.get_actor ();
      AdaptiveCriticNet3 &criticnet = *actor_critic_model.get_adaptive_critic ();

      network_learning_rates_map.at (actornet.name ())->reset ();
      network_learning_rates_map.at (criticnet.name ())->reset ();

      const vector<BaseLayer *> &actor_network_layers =
         actornet.get_network_layers ();
      for (unsigned int ndx = 0; ndx < actor_network_layers.size (); ndx++)
      {
         BaseLayer &layer = *actor_network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         string key = genkey (actornet, name);

         cout << "init " << key << endl;
         prev_actor_Hv_map[name].resize (output_sz, input_sz + 1);
         prev_actor_Hv_map[name] = 0;

         prev_actor_dEdB_map[name].resize (layer.size (), 0.0);

         prev_actor_dEdW_map[name].resize (output_sz, input_sz);
         prev_actor_dEdW_map[name] = 0;

         w_actor_map[name].resize (output_sz, input_sz + 1);
         w_actor_map[name] = 0;

         gE_td_phi[name].resize (output_sz, input_sz + 1);
         gE_td_phi[name] = 0;
      }

      const vector<BaseLayer *> &critic_network_layers =
         criticnet.get_network_layers ();

      for (unsigned int ndx = 0; ndx < critic_network_layers.size (); ndx++)
      {
         BaseLayer &layer = *critic_network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         string key = genkey (criticnet, name);

         cout << "init " << key << endl;

         prev_critic_Hv_map[name].resize (output_sz, input_sz + 1);
         prev_critic_Hv_map[name] = 0;

         prev_critic_dEdB_map[name].resize (layer.size (), 0.0);

         prev_critic_dEdW_map[name].resize (output_sz, input_sz);
         prev_critic_dEdW_map[name] = 0;

         w_critic_map[name].resize (output_sz, input_sz + 1);
         w_critic_map[name] = 0;

         gE_td_phi[name].resize (output_sz, input_sz + 1);
         gE_td_phi[name] = 0;
      }

      gE_norm = 0;
   }

   void ActorCriticTrainer3::train (EnvironmentSimulator *_trainingEnv, double _objVal)
   {
      double perf;
      double global_performance = 0.5, prev_global_performance = 9999999.0;
      double best_global_performance = 0;
      double score, prev_score = 0;
      double best_score = 0;
      double best_perf = 100.0;
      unsigned int best_epoch = 0;

      bool update_actor_flag = false;

      int consecutive_failback_count = 0;

      const TDCNeuralNet &actor = *actor_critic_model.get_actor ();
      const TDCNeuralNet &critic = *actor_critic_model.get_adaptive_critic ();

      init_train ();

      save_weights (best_weights_id);

      for (unsigned int epoch = 0; epoch < max_training_epochs; epoch++)
      {
         init_training_epoch ();

         if (actor_batch_size > 0)
            update_actor_flag =
               (epoch % actor_batch_size == 0
                || epoch == max_training_epochs - 1) ? true : false;
         else
            update_actor_flag = false;

         perf = train_exemplar (_trainingEnv, _objVal, update_actor_flag);

         // cout << endl << "----- updating critic weight -----" << endl;
         apply_delta_network_weights (*actor_critic_model.get_adaptive_critic ());
         zero_delta_network_weights (*actor_critic_model.get_adaptive_critic ());

         if (update_actor_flag)
         {
            // cout << endl << "----- updating actor weight -----" << endl;

            apply_delta_network_weights (*actor_critic_model.get_actor ());
            zero_delta_network_weights (*actor_critic_model.get_actor ());
         }

         /*
          * Get the performance of the critic network
          */
         global_performance = sim_critic (_trainingEnv, 5);

         //if (epoch % 50 == 0)
         {
            perf = 0;

            //cout << "gE_norm = " << gE_norm << endl;

            TDCNeuralNet &nn = *actor_critic_model.get_adaptive_critic ();
            const vector<BaseLayer *> network_layers = nn.get_network_layers ();
            for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
            {
               BaseLayer &layer = *network_layers[ndx];
               const string &name = layer.name ();

               unsigned int osize = layer.size ();
               unsigned int isize = layer.get_input_error ().size ();
               const Array<double> &w = w_critic_map[name];

               for (unsigned int i = 0; i < osize; i++)
                  for (unsigned int j = 0; j < isize + 1; j++)
                     perf += (gE_td_phi[name].at (i, j) / gE_norm); // * w.at(i,j);

               gE_td_phi[name] = 0;
            }

            gE_norm = 0;

            //perf = sqrt(perf);
            //global_performance = perf;
         }

         score = sim (_trainingEnv, 5);

         if (is_verbose_mode () && epoch > 0
             && (epoch < 20 || epoch % report_freq == 0))
         {
            cout << "global perf(" << epoch << ") = " << global_performance
                 << "; score = " << score << endl;
         }

         if (update_actor_flag)
         {
            if ((score - prev_score) / prev_score < -0.2)
            {
               cout << "fail back weights because of large decrease in score. "
                    << prev_score << " => " << score << endl;

               consecutive_failback_count++;
               if (consecutive_failback_count > 25)
               {
                  cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
                  break;
               }

               restore_weights (failback_weights_id);
               //restore_weights(failback_weights_id, *actor_critic_model.get_actor());
               continue;
            }
            else
            {
               prev_score = score;
               prev_global_performance = global_performance;
               consecutive_failback_count = 0;
            }

            if (score > best_score)
            {
               best_epoch = epoch;
               best_score = score;
               save_weights (best_weights_id);
            }
         }

         else // update critic
         {
            if ((global_performance - prev_global_performance)
                / prev_global_performance > 0.05 && false)
            {
               cout
                  << "fail back weights because of large increase in critic error. "
                  << prev_global_performance << " => " << global_performance
                  << endl;

               consecutive_failback_count++;
               if (consecutive_failback_count > 25)
               {
                  cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
                  //break;
                  consecutive_failback_count = 0;
                  continue;
               }

               restore_weights (failback_weights_id);
               //reduce_learning_rates(0.4);
               continue;
            }
            else
            {
               prev_global_performance = global_performance;
               prev_score = score;
               consecutive_failback_count = 0;
            }

            if (global_performance < best_perf)
            {
               best_epoch = epoch;
               best_perf = global_performance;
               save_weights (best_weights_id);
            }
         }
      }

// Restore weights to the best training epoch
      //restore_weights(best_weights_id);
      cout << ">>> best epoch = " << best_epoch << endl;
   }

   void ActorCriticTrainer3::train (EnvironmentSimulator *_trainingEnv,
                                    EnvironmentSimulator *_testEnv, double _objVal)
   {
      double perf;
      double global_performance = 0.5, prev_global_performance = 9999999.0;
      double best_global_performance = 0;
      double score, prev_score = 0;
      double best_score = 0;
      double best_perf = 100.0;
      unsigned int best_epoch = 0;

      bool update_actor_flag = false;

      int consecutive_failback_count = 0;

      const TDCNeuralNet &actor = *actor_critic_model.get_actor ();
      const TDCNeuralNet &critic = *actor_critic_model.get_adaptive_critic ();

      init_train ();

      save_weights (best_weights_id);

      for (unsigned int epoch = 0; epoch < max_training_epochs; epoch++)
      {

         /*
          * Train critic
          */
         for (unsigned int i = 0; i < critic_batch_size; i++)
         {
            init_training_epoch ();
            update_actor_flag = false;
            perf = train_exemplar (_trainingEnv, _objVal, update_actor_flag);

            apply_delta_network_weights (*actor_critic_model.get_adaptive_critic ());
            zero_delta_network_weights (*actor_critic_model.get_adaptive_critic ());

            /*
            global_performance = sim_critic(_testEnv, 350);
            score = sim(_testEnv, 350);

            if ((global_performance - prev_global_performance)
                  / prev_global_performance > 0.05 && false)
            {
               cout
                     << "fail back weights because of large increase in critic error. "
                     << prev_global_performance << " => " << global_performance
                     << endl;

               consecutive_failback_count++;
               if (consecutive_failback_count > 25)
               {
                  cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
                  //break;
                  consecutive_failback_count = 0;
                  continue;
               }

               restore_weights(failback_weights_id);
               //reduce_learning_rates(0.4);
               continue;
            }
            else
            {
               prev_global_performance = global_performance;
               prev_score = score;
               consecutive_failback_count = 0;
            }

            if (global_performance < best_perf)
            {
               best_epoch = epoch;
               best_perf = global_performance;
               save_weights(best_weights_id);
            }
            */
         }


         /*
          * Train actor
          */
         for (unsigned int i = 0; i < actor_batch_size; i++)
         {
            init_training_epoch ();
            update_actor_flag = true;
            perf = train_exemplar (_trainingEnv, _objVal, update_actor_flag);

            apply_delta_network_weights (*actor_critic_model.get_actor ());
            zero_delta_network_weights (*actor_critic_model.get_actor ());

            /*
            global_performance = sim_critic(_testEnv, 350);
            score = sim(_testEnv, 350);

            if ((score - prev_score) / prev_score < -0.2)
            {
               cout << "fail back weights because of large decrease in score. "
                     << prev_score << " => " << score << endl;

               consecutive_failback_count++;
               if (consecutive_failback_count > 25)
               {
                  cout << "Max consecutive failbacks exceeded --- EXIT" << endl;
                  break;
               }

               restore_weights(failback_weights_id);
               //restore_weights(failback_weights_id, *actor_critic_model.get_actor());
               continue;
            }
            else
            {
               prev_score = score;
               prev_global_performance = global_performance;
               consecutive_failback_count = 0;
            }

            if (score > best_score)
            {
               best_epoch = epoch;
               best_score = score;
               save_weights(best_weights_id);
            }
            */
         }




         //if (epoch % 50 == 0)
         {
            perf = 0;

            //cout << "gE_norm = " << gE_norm << endl;

            TDCNeuralNet &nn = *actor_critic_model.get_adaptive_critic ();
            const vector<BaseLayer *> network_layers = nn.get_network_layers ();
            for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
            {
               BaseLayer &layer = *network_layers[ndx];
               const string &name = layer.name ();

               unsigned int osize = layer.size ();
               unsigned int isize = layer.get_input_error ().size ();
               const Array<double> &w = w_critic_map[name];

               for (unsigned int i = 0; i < osize; i++)
                  for (unsigned int j = 0; j < isize + 1; j++)
                     perf += (gE_td_phi[name].at (i, j) / gE_norm); // * w.at(i,j);

               gE_td_phi[name] = 0;
            }

            gE_norm = 0;

            //perf = sqrt(perf);
            //global_performance = perf;
         }

         global_performance = sim_critic (_testEnv, 350);
         score = sim (_testEnv, 350);

         if (is_verbose_mode () && epoch > 0
             && (epoch < 20 || epoch % report_freq == 0))
         {
            cout << "global perf(" << epoch << ") = " << global_performance
                 << "; score = " << score << endl;
         }
      }

// Restore weights to the best training epoch
      //restore_weights(best_weights_id);
      cout << ">>> best epoch = " << best_epoch << endl;
   }

   void ActorCriticTrainer3::init_training_epoch ()
   {
      actor_critic_model.clear_error ();
      zero_delta_network_weights ();
      save_weights (failback_weights_id);

      ActorNet3 &actornet = *actor_critic_model.get_actor ();
      AdaptiveCriticNet3 &criticnet = *actor_critic_model.get_adaptive_critic ();

      const vector<BaseLayer *> &actor_network_layers =
         actornet.get_network_layers ();
      for (unsigned int ndx = 0; ndx < actor_network_layers.size (); ndx++)
      {
         BaseLayer &layer = *actor_network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         string key = genkey (actornet, name);

         prev_actor_Hv_map[name] = 0;
         prev_actor_dEdB_map[name].assign (layer.size (), 0.0);
         prev_actor_dEdW_map[name] = 0;
         gE_td_phi[name] = 0;
      }

      const vector<BaseLayer *> &critic_network_layers =
         criticnet.get_network_layers ();

      for (unsigned int ndx = 0; ndx < critic_network_layers.size (); ndx++)
      {
         BaseLayer &layer = *critic_network_layers[ndx];
         const string &name = layer.name ();
         unsigned int output_sz = layer.size ();
         unsigned int input_sz = layer.input_size ();

         string key = genkey (criticnet, name);

         prev_critic_Hv_map[name] = 0;
         prev_critic_dEdB_map[name].assign (layer.size (), 0.0);
         prev_critic_dEdW_map[name] = 0;
         gE_td_phi[name] = 0;
      }
   }

/**
 * Train one sequence/game until a terminal state is reached
 */
   double ActorCriticTrainer3::train_exemplar (EnvironmentSimulator *_env,
                                               double _objVal, bool _updateActorFlag)
   {
      Pattern actor_opatt;

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
      double td_error, actor_td_error, prev_patt_err = 0;
      double seq_sse = 0;

      vector<double> ugradient (1, 1.0);
      vector<double> zgradient (1, 0.0);

      vector<double> egradient (1);
      vector<double> opatt (1);
      vector<double> prev_opatt (1);
      vector<double> pgrad (1);
      vector<double> tgt_patt (1);

      bool saved_stochastic_flag = actor_critic_model.get_stochastic_action ();
      double saved_var =
         actor_critic_model.get_actor ()->get_stochastic_action_var ();

      //actor_critic_model.get_actor()->set_stochastic_action_gain(4.0);

      /******************************************************************
       * Set up model, state, and training data on initial state for
       * training on subsequent states
       *
       */

// Don't use stochastic activation if we are training the actor
//   if (_updateActorFlag)
//      actor_critic_model.set_stochastic_action(false);
// Reset the environment to start a new "game"
      state = _env->reset ();

// Initial activation of the model with the start state
      ac_out = actor_critic_model (state);
      opatt[0] = ac_out.reinforcement ();

      actor_critic_model.clear_error ();
      actor_critic_model.backprop (ugradient);

      if (print_gradient)
      {
         cout << "***** initial activation *****" << endl;
         for (unsigned int i = 0; i < state ().size (); i++)
            cout << state ().at (i) << " ";
         cout << " => ";
         cout << "critic rsig " << ac_out.reinforcement () << endl;

         cout << "\nactor net raw output" << endl;
         actor_opatt = actor_critic_model.get_actor ()->raw ();
         for (unsigned int i = 0; i < actor_opatt ().size (); i++)
            cout << actor_opatt ().at (i) << " ";
         cout << " =>  " << ac_out.action ().name () << endl;
      }

      step_no++;

      /***************************************************
       * Train on subsequent states
       */
      do
      {
         actor_critic_model.set_print_gradient (false);

         // Save current input
         prev_state = state;

         // Update the environment with the recommended action
         state = _env->next_state (ac_out.action ());

         // Check for external reinforcement signal
         external_rsig = _env->get_reinforcement (external_rflag);

         // Activate the model with the current state vector
         prev_ac_out = ac_out;

         prev_opatt = opatt;

         if (_updateActorFlag)
         {
            save_actor_gradient ();
            calc_actor_adj2 (*actor_critic_model.get_actor (), (1.0 - opatt.at (0)));
         }

         // Save gradient info for critic and actor networks
         save_critic_gradient ();

         // TD2Trainer presents the next state even if it's terminal
         // Dunno why but try it here
         ac_out = actor_critic_model (state);
         opatt[0] = ac_out.reinforcement ();

         actor_critic_model.clear_error ();
         actor_critic_model.set_print_gradient (print_gradient);
         actor_critic_model.backprop (ugradient);
         actor_critic_model.set_print_gradient (false);

         if (!_env->is_terminal_state (state))
         {
            if (print_gradient)
            {
               cout << "***** next activation *****" << endl;
               for (unsigned int i = 0; i < state ().size (); i++)
                  cout << state ().at (i) << " ";
               cout << " => ";
               cout << "critic rsig " << ac_out.reinforcement () << endl;

               cout << "actor net raw output" << endl;
               actor_opatt = actor_critic_model.get_actor ()->raw ();
               for (unsigned int i = 0; i < actor_opatt ().size (); i++)
                  cout << actor_opatt ().at (i) << " ";
               cout << " =>  " << ac_out.action ().name () << endl;
            }

            /*
             * Use the external or internal reinforcement signal to calculate
             * the error gradient we will backprop through the actor-critic
             * network in order to train the critic
             */
            training_rsig =
               (external_rflag) ? external_rsig : ac_out.reinforcement ();

            if (print_gradient)
               if (external_rflag)
                  cout << "tsig = external non-term " << training_rsig << endl;
               else
                  cout << "tsig = internal non-term " << training_rsig << endl;
         }
         else
         {
            training_rsig = external_rsig;

            /*
            actor_critic_model.clear_error();
            actor_critic_model.set_print_gradient(print_gradient);
            actor_critic_model.backprop(zgradient);
            actor_critic_model.set_print_gradient(false);
            */

            if (print_gradient)
            {
               cout << "***** terminal *****" << endl;
               for (unsigned int i = 0; i < state ().size (); i++)
                  cout << state ().at (i) << " ";
               cout << endl;
               cout << "tsig = external terminal " << training_rsig << endl;
            }
         }

         /*
          * Temporal difference learning needs to save the old network
          * weight gradients
          */

         // Adaptive critic can only learn after the first step
         if (step_no > 0)
         {
            // TODO - activate network for previous input state
            //prev_ac_out = actor_critic_model(prev_state);
            prev_opatt[0] = prev_ac_out.reinforcement ();

            if (predict_mode == FINAL_COST)
            {
               tgt_patt[0] = training_rsig;
               error_func (patt_sse, egradient, prev_opatt, tgt_patt);

               if (print_gradient)
               {
                  cout << "tgt patt = " << tgt_patt[0] << endl;
                  cout << "prev opatt = " << prev_opatt[0] << endl;
               }
            }
            else if (predict_mode == CUMULATIVE_COST)
            {

               if (_env->is_terminal_state (state))
                  td_error = external_rsig - prev_opatt.at (0);
               else
                  td_error = external_rsig + gamma * opatt.at (0)
                             - prev_opatt.at (0);

               egradient[0] = td_error;
               patt_sse = 0.5 * (td_error * td_error);
            }

            actor_td_error = 1.0 - prev_opatt.at (0);
            //seq_sse += patt_sse;

            double save_gamma = gamma;
            if (_env->is_terminal_state (state))
               gamma = 0.0;

            calc_critic_adj (*actor_critic_model.get_adaptive_critic (), td_error);

            if (_env->is_terminal_state (state))
               gamma = save_gamma;

            /*
             * Calculate the weight updates for the PREVIOUS network activation
             */
            TDCNeuralNet &nn = *actor_critic_model.get_adaptive_critic ();
            const vector<BaseLayer *> network_layers = nn.get_network_layers ();
            for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
            {
               BaseLayer &layer = *network_layers[ndx];
               const string &name = layer.name ();

               unsigned int osize = layer.size ();
               unsigned int isize = layer.get_input_error ().size ();

               const Array<double> &prev_dEdW = prev_critic_dEdW_map[name];
               const Array<double> &w = w_critic_map[name];
               nn.set_v (name, w);

               for (unsigned int i = 0; i < osize; i++)
               {
                  gE_td_phi[name].at (i, 0) += td_error;
                  for (unsigned int j = 0; j < isize; j++)
                     gE_td_phi[name].at (i, j + 1) += td_error * prev_dEdW.at (i, j);
               }
            }

            if (print_gradient)
               cout << "critic egradient = " << egradient[0] << endl;
         }

         gE_norm++;
         step_no++;

      }
      while (!_env->is_terminal_state (state));

      seq_sse = external_rsig;

// Restore model stochastic action flag setting
      actor_critic_model.set_stochastic_action (saved_stochastic_flag);
      actor_critic_model.get_actor ()->set_stochastic_action_var (saved_var);

      return seq_sse;
   }

/**
 * Train one sequence/game until a terminal state is reached
 */
   double ActorCriticTrainer3::train_exemplar2 (EnvironmentSimulator *_env,
                                                double _objVal, bool _updateActorFlag)
   {
      Pattern actor_opatt;

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
      double td_error, prev_patt_err = 0;
      double seq_sse = 0;

      vector<double> ugradient (1, 1.0);

      vector<double> egradient (1);
      vector<double> prev_opatt (1);
      vector<double> tgt_patt (1);

      bool saved_stochastic_flag = actor_critic_model.get_stochastic_action ();
      double saved_var =
         actor_critic_model.get_actor ()->get_stochastic_action_var ();

      //actor_critic_model.get_actor()->set_stochastic_action_gain(4.0);

      /******************************************************************
       * Set up model, state, and training data on initial state for
       * training on subsequent states
       *
       */

// Don't use stochastic activation if we are training the actor
//   if (_updateActorFlag)
//      actor_critic_model.set_stochastic_action(false);
// Reset the environment to start a new "game"
      state = _env->reset ();

// Initial activation of the model with the start state
      ac_out = actor_critic_model (state, _env->hint ());

      if (print_gradient)
      {
         cout << "***** initial activation *****" << endl;
         for (unsigned int i = 0; i < state ().size (); i++)
            cout << state ().at (i) << " ";
         cout << " => ";
         cout << "critic rsig " << ac_out.reinforcement () << endl;

         cout << "\nactor net raw output" << endl;
         actor_opatt = actor_critic_model.get_actor ()->raw ();
         for (unsigned int i = 0; i < actor_opatt ().size (); i++)
            cout << actor_opatt ().at (i) << " ";
         cout << " =>  " << ac_out.action ().name () << endl;
      }

      /*
       * Temporal difference learning needs to save the old network
       * weight gradients and add the new weight gradients to the
       * discounted sum of the weight gradients
       */

      actor_critic_model.clear_error ();
      actor_critic_model.set_print_gradient (print_gradient);
      actor_critic_model.backprop (ugradient);
      actor_critic_model.set_print_gradient (false);

      step_no++;

      /***************************************************
       * Train on subsequent states
       */
      do
      {
         // Save current input
         prev_state = state;

         // Update the environment with the recommended action
         state = _env->next_state (ac_out.action ());

         // Check for external reinforcement signal
         external_rsig = _env->get_reinforcement (external_rflag);

         // Activate the model with the current state vector
         prev_ac_out = ac_out;

         // Save gradient info for critic and actor networks
         save_critic_gradient ();
         save_actor_gradient ();

         // TD2Trainer presents the next state even if it's terminal
         // Dunno why but try it here
         ac_out = actor_critic_model (state, _env->hint ());

         if (!_env->is_terminal_state (state))
         {
            //ac_out = actor_critic_model(state);

            if (print_gradient)
            {
               cout << "***** next activation *****" << endl;
               for (unsigned int i = 0; i < state ().size (); i++)
                  cout << state ().at (i) << " ";
               cout << " => ";
               cout << "critic rsig " << ac_out.reinforcement () << endl;

               cout << "actor net raw output" << endl;
               actor_opatt = actor_critic_model.get_actor ()->raw ();
               for (unsigned int i = 0; i < actor_opatt ().size (); i++)
                  cout << actor_opatt ().at (i) << " ";
               cout << " =>  " << ac_out.action ().name () << endl;
            }

            /*
             * Use the external or internal reinforcement signal to calculate
             * the error gradient we will backprop through the actor-critic
             * network in order to train the critic
             */
            training_rsig =
               (external_rflag) ? external_rsig : ac_out.reinforcement ();

            if (print_gradient)
               if (external_rflag)
                  cout << "tsig = external non-term " << training_rsig << endl;
               else
                  cout << "tsig = internal non-term " << training_rsig << endl;
         }
         else
         {

            training_rsig = external_rsig;

            if (print_gradient)
            {
               cout << "***** terminal *****" << endl;
               for (unsigned int i = 0; i < state ().size (); i++)
                  cout << state ().at (i) << " ";
               cout << endl;
               cout << "tsig = external terminal " << training_rsig << endl;
            }
         }

         /*
          * Temporal difference learning needs to save the old network
          * weight gradients
          */

         actor_critic_model.clear_error ();
         actor_critic_model.set_print_gradient (print_gradient);
         actor_critic_model.backprop (ugradient);
         actor_critic_model.set_print_gradient (false);

         // Adaptive critic can only learn after the first step
         if (step_no > 0)
         {
            // TODO - activate network for previous input state
            //prev_ac_out = actor_critic_model(prev_state);
            prev_opatt[0] = prev_ac_out.reinforcement ();

            if (predict_mode == FINAL_COST)
            {
               tgt_patt[0] = training_rsig;
               error_func (patt_sse, egradient, prev_opatt, tgt_patt);

               if (print_gradient)
               {
                  cout << "tgt patt = " << tgt_patt[0] << endl;
                  cout << "prev opatt = " << prev_opatt[0] << endl;
               }
            }
            else if (predict_mode == CUMULATIVE_COST)
            {

               if (_env->is_terminal_state (state))
                  td_error = external_rsig - prev_opatt.at (0);
               else
                  td_error = external_rsig + gamma * ac_out.reinforcement ()
                             - prev_opatt.at (0);

               egradient[0] = td_error;
               patt_sse = 0.5 * (td_error * td_error);
            }
            //seq_sse += patt_sse;

            calc_critic_adj (*actor_critic_model.get_adaptive_critic (), td_error);

            /*
             * Calculate the weight updates for the PREVIOUS network activation
             */
            TDCNeuralNet &nn = *actor_critic_model.get_adaptive_critic ();
            const vector<BaseLayer *> network_layers = nn.get_network_layers ();
            for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
            {
               BaseLayer &layer = *network_layers[ndx];
               const string &name = layer.name ();

               unsigned int osize = layer.size ();
               unsigned int isize = layer.get_input_error ().size ();

               const Array<double> &prev_dEdW = prev_critic_dEdW_map[name];
               const Array<double> &w = w_critic_map[name];
               nn.set_v (name, w);

               for (unsigned int i = 0; i < osize; i++)
               {
                  gE_td_phi[name].at (i, 0) += td_error;
                  for (unsigned int j = 0; j < isize; j++)
                     gE_td_phi[name].at (i, j + 1) += td_error * prev_dEdW.at (i, j);
               }
            }

            if (print_gradient)
               cout << "critic egradient = " << egradient[0] << endl;

            // !!!!! Oooops. Actor can only learn after the 1st step as well???
            if (_updateActorFlag)
            {
               prev_opatt[0] = prev_ac_out.reinforcement ();

               /*
                * Use the ultimate objective value to calculate the error gradient
                * we will backprop through the actor-critic network in order to train
                * the actor network
                */
               tgt_patt[0] = _objVal;
               error_func (patt_sse, egradient, prev_opatt, tgt_patt);

               if (print_gradient)
                  cout << "actor egradient = " << egradient[0] << endl;

               /*
                network_learning_rates_map.at(actor_critic_model.get_actor()->name())->update_learning_rate_adjustments();
                */

               calc_actor_adj (*actor_critic_model.get_actor (), td_error);
            }
         }

         gE_norm++;
         step_no++;

      }
      while (!_env->is_terminal_state (state));
//seq_sse = (step_no > 0) ? seq_sse / step_no : 0;
      seq_sse = external_rsig;

// Restore model stochastic action flag setting
      actor_critic_model.set_stochastic_action (saved_stochastic_flag);
      actor_critic_model.get_actor ()->set_stochastic_action_var (saved_var);

      return seq_sse;
   }

   void ActorCriticTrainer3::calc_critic_adj (const TDCNeuralNet &_net,
                                              double _tdErr)
   {
      NetworkWeightsData &network_deltas = get_cached_network_weights (
         delta_network_weights_id, _net);

      /*
       * Calculate phi * w
       */
      double phi_w = 0;

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         const vector<double> &prev_dEdB = prev_critic_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_critic_dEdW_map[name];
         const Array<double> &w = w_critic_map[name];

         for (unsigned int rowndx = 0; rowndx < prev_dEdW.rowDim (); rowndx++)
         {
            phi_w += prev_dEdB.at (rowndx) * w.at (rowndx, 0);
            for (unsigned int colndx = 0; colndx < prev_dEdW.colDim (); colndx++)
               phi_w += prev_dEdW.at (rowndx, colndx) * w.at (rowndx, colndx + 1);
         }
      }

      double h;

      double hb;
      double theta_p_b;
      double theta_p;

      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int osize = layer.size ();
         unsigned int isize = layer.input_size ();

         // Get the learning rates for the biases
         vector<double> layer_bias_lr =
            network_learning_rates_map.at (_net.name ())->get_bias_learning_rates ().at (
               layer.name ());

         // Get the learning rates for the weights
         Array<double> layer_weights_lr = network_learning_rates_map.at (
            _net.name ())->get_weight_learning_rates ().at (layer.name ());

         const Array<double> &Hv = prev_critic_Hv_map.at (layer.name ());

         const vector<double> &dEdB = layer.get_dEdB ();
         const Array<double> &dEdW = layer.get_dEdW ();

         const vector<double> &prev_dEdB = prev_critic_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_critic_dEdW_map[name];
         const Array<double> &w = w_critic_map[name];

         LayerWeightsData &layer_deltas = network_deltas.layer_weights (name);

         for (unsigned int oNdx = 0; oNdx < osize; oNdx++)
         {
            for (unsigned int iNdx = 0; iNdx < isize; iNdx++)
            {

               h = (_tdErr - phi_w) * Hv.at (oNdx, iNdx);

               theta_p = _tdErr * prev_dEdW.at (oNdx, iNdx)
                         - gamma * dEdW.at (oNdx, iNdx) * phi_w - h;

               layer_deltas.weights.at (oNdx, iNdx) += layer_weights_lr.at (oNdx,
                                                                            iNdx) * theta_p;
            }

            hb = (_tdErr - phi_w) * Hv.at (oNdx, 0);
            theta_p_b = _tdErr * prev_dEdB.at (oNdx) - gamma * dEdB.at (oNdx) * phi_w
                        - hb;
            layer_deltas.biases.at (oNdx) += layer_bias_lr.at (oNdx) * theta_p_b;
         }
      } // end loop through layers

      /*
       * Update w estimate
       */
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int osize = layer.size ();
         unsigned int isize = layer.input_size ();

         // Get the learning rates for the biases
         vector<double> layer_bias_lr =
            network_learning_rates_map.at (_net.name ())->get_bias_learning_rates ().at (
               layer.name ());

         // Get the learning rates for the weights
         Array<double> layer_weights_lr = network_learning_rates_map.at (
            _net.name ())->get_weight_learning_rates ().at (layer.name ());

         const vector<double> &prev_dEdB = prev_critic_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_critic_dEdW_map[name];
         Array<double> &w = w_critic_map[name];

         for (unsigned int oNdx = 0; oNdx < osize; oNdx++)
         {
            w.at (oNdx, 0) = w.at (oNdx, 0)
                             + slow_lr_multiplier * layer_bias_lr.at (oNdx) * (_tdErr - phi_w)
                               * prev_dEdB.at (oNdx);

            for (unsigned int iNdx = 0; iNdx < isize; iNdx++)
            {
               w.at (oNdx, iNdx + 1) = w.at (oNdx, iNdx + 1)
                                       + slow_lr_multiplier * layer_weights_lr.at (oNdx, iNdx)
                                         * (_tdErr - phi_w) * prev_dEdW.at (oNdx, iNdx);
            }
         }
      } // end loop through layers
   }

   void ActorCriticTrainer3::calc_actor_adj (const TDCNeuralNet &_net,
                                             double _tdErr)
   {
      NetworkWeightsData &network_deltas = get_cached_network_weights (
         delta_network_weights_id, _net);

      /*
       * Calculate phi * w
       */
      double phi_w = 0;

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         const vector<double> &prev_dEdB = prev_actor_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_actor_dEdW_map[name];
         const Array<double> &w = w_actor_map[name];

         for (unsigned int rowndx = 0; rowndx < prev_dEdW.rowDim (); rowndx++)
         {
            phi_w += prev_dEdB.at (rowndx) * w.at (rowndx, 0);
            for (unsigned int colndx = 0; colndx < prev_dEdW.colDim (); colndx++)
               phi_w += prev_dEdW.at (rowndx, colndx) * w.at (rowndx, colndx + 1);
         }
      }

      double h;

      double hb;
      double theta_p_b;
      double theta_p;

      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int osize = layer.size ();
         unsigned int isize = layer.input_size ();

         // Get the learning rates for the biases
         vector<double> layer_bias_lr =
            network_learning_rates_map.at (_net.name ())->get_bias_learning_rates ().at (
               layer.name ());

         // Get the learning rates for the weights
         Array<double> layer_weights_lr = network_learning_rates_map.at (
            _net.name ())->get_weight_learning_rates ().at (layer.name ());

         const Array<double> &Hv = prev_actor_Hv_map.at (layer.name ());

         const vector<double> &dEdB = layer.get_dEdB ();
         const Array<double> &dEdW = layer.get_dEdW ();

         const vector<double> &prev_dEdB = prev_actor_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_actor_dEdW_map[name];
         const Array<double> &w = w_actor_map[name];

         LayerWeightsData &layer_deltas = network_deltas.layer_weights (name);

         for (unsigned int oNdx = 0; oNdx < osize; oNdx++)
         {
            for (unsigned int iNdx = 0; iNdx < isize; iNdx++)
            {

               h = (_tdErr - phi_w) * Hv.at (oNdx, iNdx);

               theta_p = _tdErr * prev_dEdW.at (oNdx, iNdx)
                         - gamma * dEdW.at (oNdx, iNdx) * phi_w - h;

               layer_deltas.weights.at (oNdx, iNdx) += layer_weights_lr.at (oNdx,
                                                                            iNdx) * theta_p;
            }

            hb = (_tdErr - phi_w) * Hv.at (oNdx, 0);
            theta_p_b = _tdErr * prev_dEdB.at (oNdx) - gamma * dEdB.at (oNdx) * phi_w
                        - hb;
            layer_deltas.biases.at (oNdx) += layer_bias_lr.at (oNdx) * theta_p_b;
         }
      } // end loop through layers

      /*
       * Update w estimate
       */
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();
         unsigned int osize = layer.size ();
         unsigned int isize = layer.input_size ();

         // Get the learning rates for the biases
         vector<double> layer_bias_lr =
            network_learning_rates_map.at (_net.name ())->get_bias_learning_rates ().at (
               layer.name ());

         // Get the learning rates for the weights
         Array<double> layer_weights_lr = network_learning_rates_map.at (
            _net.name ())->get_weight_learning_rates ().at (layer.name ());

         const vector<double> &prev_dEdB = prev_actor_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_actor_dEdW_map[name];
         Array<double> &w = w_actor_map[name];

         for (unsigned int oNdx = 0; oNdx < osize; oNdx++)
         {
            w.at (oNdx, 0) = w.at (oNdx, 0)
                             + slow_lr_multiplier * layer_bias_lr.at (oNdx) * (_tdErr - phi_w)
                               * prev_dEdB.at (oNdx);

            for (unsigned int iNdx = 0; iNdx < isize; iNdx++)
            {
               w.at (oNdx, iNdx + 1) = w.at (oNdx, iNdx + 1)
                                       + slow_lr_multiplier * layer_weights_lr.at (oNdx, iNdx)
                                         * (_tdErr - phi_w) * prev_dEdW.at (oNdx, iNdx);
            }
         }
      } // end loop through layers
   }

   void ActorCriticTrainer3::calc_actor_adj2 (const TDCNeuralNet &_net, double _Err)
   {
      if (print_gradient)
      {
         cout << "  ******* actor U-J = " << _Err << endl;
      }

      NetworkWeightsData &network_deltas = get_cached_network_weights (
         delta_network_weights_id, _net);

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         const vector<double> &prev_dEdB = prev_actor_dEdB_map[name];
         const Array<double> &prev_dEdW = prev_actor_dEdW_map[name];

         // Get the learning rates for the biases
         vector<double> layer_bias_lr =
            network_learning_rates_map.at (_net.name ())->get_bias_learning_rates ().at (
               layer.name ());

         // Get the learning rates for the weights
         Array<double> layer_weights_lr = network_learning_rates_map.at (
            _net.name ())->get_weight_learning_rates ().at (layer.name ());

         LayerWeightsData &layer_deltas = network_deltas.layer_weights (name);

         for (unsigned int netin_ndx = 0; netin_ndx < prev_dEdB.size (); netin_ndx++)
            layer_deltas.biases.at (netin_ndx) += layer_bias_lr[netin_ndx] * _Err
                                                  * prev_dEdB.at (netin_ndx);

         unsigned int layer_input_size = layer.input_size ();
         for (unsigned int out_ndx = 0; out_ndx < layer.size (); out_ndx++)
         {
            for (unsigned int in_ndx = 0; in_ndx < layer_input_size; in_ndx++)
            {
               layer_deltas.weights.at (out_ndx, in_ndx) +=
                  layer_weights_lr[out_ndx][in_ndx] * _Err * prev_dEdW.at (out_ndx, in_ndx);
            }
         }
      }
   }

   void ActorCriticTrainer3::save_weights (const string &_id)
   {
      save_weights (_id, *actor_critic_model.get_actor ());
      save_weights (_id, *actor_critic_model.get_adaptive_critic ());
   }

   void ActorCriticTrainer3::save_weights (const string &_id,
                                           const TDCNeuralNet &_net)
   {
// TODO - check the buffer id to make sure it's not one of the reserved values
      NetworkWeightsData &network_weights_data = get_cached_network_weights (_id,
                                                                             _net);

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         string name = layer.name ();

         LayerWeightsData &layer_weights_data = network_weights_data.layer_weights (
            name);

         // BaseLayer value at time step 0 is the initial layer value
         layer_weights_data.initial_value = layer (0);

         layer_weights_data.biases = layer.get_biases ();
         layer_weights_data.weights = layer.get_weights ();
      }
   }

   void ActorCriticTrainer3::restore_weights (const string &_id)
   {
      restore_weights (_id, *actor_critic_model.get_actor ());
      restore_weights (_id, *actor_critic_model.get_adaptive_critic ());
   }

   void ActorCriticTrainer3::restore_weights (const string &_id,
                                              const TDCNeuralNet &_net)
   {
// TODO - check the buffer id to make sure it's not one of the reserved values
      string key = genkey (_net, _id);
      NetworkWeightsData &network_weights_data = network_weights_cache[key];

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         string name = layer.name ();

         LayerWeightsData &layer_weights_data = network_weights_data.layer_weights (
            name);

         layer.set_biases (layer_weights_data.biases);
         layer.set_weights (layer_weights_data.weights);
      }
   }

   void ActorCriticTrainer3::alloc_network_weights_cache_entry (const string &_id)
   {
      alloc_network_weights_cache_entry (_id, *actor_critic_model.get_actor ());
      alloc_network_weights_cache_entry (_id,
                                         *actor_critic_model.get_adaptive_critic ());
   }

   void ActorCriticTrainer3::alloc_network_weights_cache_entry (const string &_id,
                                                                const TDCNeuralNet &_net)
   {
      string key = genkey (_net, _id);
      NetworkWeightsData &network_weights = network_weights_cache[key];

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         const string &name = layer.name ();

         const Array<double> &layer_weights = layer.get_weights ();
         unsigned int rows = layer_weights.rowDim ();
         unsigned int cols = layer_weights.colDim ();

         LayerWeightsData &layer_weights_data = network_weights.new_layer_weights (
            name);

         layer_weights_data.initial_value.resize (layer.size (), 0.0);
         layer_weights_data.biases.resize (layer.get_biases ().size (), 0.0);

         layer_weights_data.weights.resize (rows, cols);
         layer_weights_data.weights = 0;
      }
   }

   void ActorCriticTrainer3::apply_delta_network_weights ()
   {
      apply_delta_network_weights (*actor_critic_model.get_actor ());
      apply_delta_network_weights (*actor_critic_model.get_adaptive_critic ());
   }

   void ActorCriticTrainer3::apply_delta_network_weights (TDCNeuralNet &_net)
   {

      if (print_gradient)
         cout << " *** apply delta weights *****" << endl;

      NetworkWeightsData &delta_network_weights = get_cached_network_weights (
         delta_network_weights_id, _net);
      NetworkWeightsData &prev_delta_network_weights = get_cached_network_weights (
         previous_delta_network_weights_id, _net);
      NetworkWeightsData &adjusted_network_weights = get_cached_network_weights (
         adjusted_network_weights_id, _net);

      const vector<BaseLayer *> network_layers = _net.get_network_layers ();
      for (unsigned int ndx = 0; ndx < network_layers.size (); ndx++)
      {
         BaseLayer &layer = *network_layers[ndx];
         string name = layer.name ();

         if (print_gradient)
            cout << "============= " << _net.name () << " " << layer.name ()
                 << " ==============" << endl;

         const LayerWeightsData &delta_layer_weights_data =
            delta_network_weights.layer_weights (name);
         const LayerWeightsData &prev_delta_layer_weights_data =
            prev_delta_network_weights.layer_weights (name);
         LayerWeightsData &adjusted_layer_weights_data =
            adjusted_network_weights.layer_weights (name);

         const vector<double> &delta_layer_biases = delta_layer_weights_data.biases;
         const vector<double> &prev_delta_layer_biases =
            prev_delta_layer_weights_data.biases;
         vector<double> &adjusted_layer_biases = adjusted_layer_weights_data.biases;

         if (print_gradient)
            cout << "   biases : " << endl;

         adjusted_layer_biases = layer.get_biases ();
         for (unsigned int ndx = 0; ndx < adjusted_layer_biases.size (); ndx++)
         {
            adjusted_layer_biases.at (ndx) += learning_momentum
                                              * prev_delta_layer_biases.at (ndx) + delta_layer_biases.at (ndx);

            if (print_gradient)
            {
               if (ndx > 0)
                  cout << ", ";
               cout << delta_layer_biases.at (ndx);
            }
         }
         if (print_gradient)
            cout << endl << "-------------" << endl;

         if (layer.is_learn_biases ())
            layer.set_biases (adjusted_layer_biases);

         const Array<double> &delta_layer_weights =
            delta_layer_weights_data.weights;
         const Array<double> &prev_delta_layer_weights =
            prev_delta_layer_weights_data.weights;
         Array<double> &adjusted_layer_weights =
            adjusted_layer_weights_data.weights;

         adjusted_layer_weights = layer.get_weights ();

         unsigned int row_sz = adjusted_layer_weights.rowDim ();
         unsigned int col_sz = adjusted_layer_weights.colDim ();

         if (print_gradient)
            cout << "   weights : " << endl;

         for (unsigned int row = 0; row < row_sz; row++)
         {
            for (unsigned int col = 0; col < col_sz; col++)
            {
               adjusted_layer_weights.at (row, col) += learning_momentum
                                                       * prev_delta_layer_weights.at (row, col)
                                                       + delta_layer_weights.at (row, col);

               if (print_gradient)
               {
                  if (col > 0)
                     cout << ", ";
                  cout << delta_layer_weights.at (row, col);
               }
            }

            if (print_gradient)
               cout << endl;
         }

         if (layer.is_learn_weights ())
            layer.set_weights (adjusted_layer_weights);
      }

      if (print_gradient)
         cout << "-------------" << endl;

// TODO - decide if we should save off the current delta weights here
      prev_delta_network_weights = delta_network_weights;
   }

   void ActorCriticTrainer3::zero_delta_network_weights ()
   {
      zero_delta_network_weights (*actor_critic_model.get_actor ());
      zero_delta_network_weights (*actor_critic_model.get_adaptive_critic ());
   }

   void ActorCriticTrainer3::zero_delta_network_weights (const TDCNeuralNet &_net)
   {
// Get a reference from the buffer to the entry containing the delta
// network weights
      NetworkWeightsData &delta_network_weights = get_cached_network_weights (
         delta_network_weights_id, _net);

// Iterate through the weights for each layer

      const set<string> &key_set = delta_network_weights.keySet ();
      set<string>::iterator iter;
      for (iter = key_set.begin (); iter != key_set.end (); iter++)
      {
         const string &key_str = *iter;
         LayerWeightsData &layer_weights_data =
            delta_network_weights.layer_weights (key_str);

         unsigned int sz;

         // Clear deltas for initial layer value
         sz = layer_weights_data.initial_value.size ();
         for (unsigned int ndx = 0; ndx < sz; ndx++)
            layer_weights_data.initial_value[ndx] = 0;

         // Clear deltas for layer biases
         sz = layer_weights_data.biases.size ();
         for (unsigned int ndx = 0; ndx < sz; ndx++)
            layer_weights_data.biases[ndx] = 0;

         // Clear deltas for layer weights
         layer_weights_data.weights = 0;
      }
   }

   int ActorCriticTrainer3::urand (int n)
   {
      int top = ((((RAND_MAX - n) + 1) / n) * n - 1) + n;
      int r;
      do
      {
         r = rand ();
      }
      while (r > top);
      return (r % n);
   }

   void ActorCriticTrainer3::alloc_network_learning_rates ()
   {
      const TDCNeuralNet &actor = *actor_critic_model.get_actor ();
      const TDCNeuralNet &critic = *actor_critic_model.get_adaptive_critic ();

      if (network_learning_rates_map.find (actor.name ())
          == network_learning_rates_map.end ())
         network_learning_rates_map[actor.name ()] = new ConstantLearningRate (
            actor);

      if (network_learning_rates_map.find (critic.name ())
          == network_learning_rates_map.end ())
         network_learning_rates_map[critic.name ()] = new ConstantLearningRate (
            critic);
   }

   void ActorCriticTrainer3::reduce_learning_rates (double val)
   {
      const TDCNeuralNet &actor = *actor_critic_model.get_actor ();
      const TDCNeuralNet &critic = *actor_critic_model.get_adaptive_critic ();

      if (network_learning_rates_map.find (actor.name ())
          != network_learning_rates_map.end ())
         network_learning_rates_map[actor.name ()]->reduce_learning_rate (val);

      if (network_learning_rates_map.find (critic.name ())
          != network_learning_rates_map.end ())
         network_learning_rates_map[critic.name ()]->reduce_learning_rate (val);

   }
} /* namespace flexnnet */
