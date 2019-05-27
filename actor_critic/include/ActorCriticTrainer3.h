/*
 * ActorCriticTrainer3.h
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ACTORCRITIC_TRAINER3_H_
#define FLEX_NEURALNET_ACTORCRITIC_TRAINER3_H_

#include <vector>
#include <string>

#include "ActorCriticNet3.h"
#include "EnvironmentSimulator.h"
#include "TrainingRecord.h"
#include "NetworkWeightsData.h"
#include "DeltaBarDeltaLearningRate.h"
#include "ConstantLearningRate.h"

using namespace std;

namespace flexnnet
{

   class ActorCriticTrainer3
   {
   public:
      ActorCriticTrainer3 (ActorCriticNet3 &_model);
      virtual ~ActorCriticTrainer3 ();

      /* ***************************************************
       *    Public setter methods
       */
      void set_max_epochs (long _epochs);
      void set_lambda (double l);
      void set_gamma (double l);
      void set_global_learning_rate (double _rate);
      void set_slow_learning_rate_multiplier (double _val);
      void set_actor_learning_rate (double _rate);
      void set_critic_learning_rate (double _rate);
      void set_learning_momentum (double _rate);
      void set_performance_goal (double _perf);
      void set_min_gradient (double _min);
      void set_max_validation_failures (int _count);

      void set_batch_mode ();
      void set_minibatch_mode (unsigned int _size);
      void set_online_mode ();

      void set_random_order_mode (bool _val);
      void set_report_frequency (int _freq);
      void set_verbose (bool _mode);

      void set_critic_batch_size (unsigned int _sz);
      void set_actor_batch_size (unsigned int _sz);

      void set_print_gradient (bool _val);

      /* ****************************************************
       *    Public getter methods
       */

      long max_epochs ();
      double performance_goal ();
      double min_gradient ();
      int max_validation_failures ();

      bool is_batch_mode ();
      bool is_online_mode ();
      bool is_minibatch_mode ();

      void set_predict_final_cost ();
      void set_predict_cumulative_cost ();

      bool is_random_order_mode ();
      bool is_verbose_mode ();
      unsigned int minibatch_size ();
      int report_frequency ();

      double global_perf ();

      const TrainingRecord &training_record ();

      /*
       * Activate the model for the specified environment
       */
      virtual double sim (EnvironmentSimulator *_env, unsigned int _sampleCount);
      virtual double sim_critic (EnvironmentSimulator *_env, unsigned int _sampleCount);

      virtual double sim2 (EnvironmentSimulator *_env, unsigned int _sampleCount);
      virtual double sim_critic2 (EnvironmentSimulator *_env, unsigned int _sampleCount);

      virtual void train (EnvironmentSimulator *_trainingEnvSet, double _objVal);
      virtual void train (EnvironmentSimulator *_trainingEnvSet, EnvironmentSimulator *_testEnvSet, double _objVal);

   private:
      void init_train ();
      void init_training_epoch ();
      virtual double train_exemplar (EnvironmentSimulator *_trainingEnv, double _objVal, bool _updateActorFlag);
      virtual double train_exemplar2 (EnvironmentSimulator *_trainingEnv, double _objVal, bool _updateActorFlag);

      virtual void calc_critic_adj (const TDCNeuralNet &_net, double _tdError);
      virtual void calc_actor_adj (const TDCNeuralNet &_net, double _tdErr);
      virtual void calc_actor_adj2 (const TDCNeuralNet &_net, double _Err);
      virtual void activate_actor_critic (const Pattern &_currStateVec, const Pattern &_actionVec);

      void reduce_learning_rates (double val);

   protected:

      // Save gradient info for critic and actor networks
      void save_critic_gradient ();
      void save_actor_gradient ();

      /* ****************************************************
       *    Protected methods to save and restore network
       *    network weights
       */
      void save_weights (const string &_id);
      void save_weights (const string &_id, const TDCNeuralNet &_net);

      void restore_weights (const string &_id);
      void restore_weights (const string &_id, const TDCNeuralNet &_net);

      /* *****************************************************
       *    Protected helper methods for managing network
       *    weight training deltas and other weights cache
       */
      bool initialized;
      NetworkWeightsData &get_cached_network_weights (const string &_id, const TDCNeuralNet &_net);

      void alloc_network_weights_cache_entry (
         const string &_id);
      void alloc_network_weights_cache_entry (
         const string &_id, const TDCNeuralNet &_net);

      void apply_delta_network_weights ();
      void apply_delta_network_weights (TDCNeuralNet &_net);

      void zero_delta_network_weights ();
      void zero_delta_network_weights (const TDCNeuralNet &_net);

      void alloc_network_learning_rates ();

      /* *******************************************************
       *    Protected methods to manage performance trace
       */
      void clear_training_record ();
      void update_training_record (unsigned int _epoch, unsigned int _stopSig, const TrainingRecord::Entry &_trEntry);

      /* *******************************************************
       *    Protected utility functions
       */
      int urand (int n);
      const string &genkey (const TDCNeuralNet &_net, const string &_id);

   protected:

      /* ***********************************************
       *    static const data member
       */
      static const long default_max_epochs;
      static const double default_learning_momentum;
      static const double default_performance_goal;
      static const double default_min_gradient;
      static const int default_max_valid_fail;
      static const bool default_batch_mode;
      static const unsigned int default_batch_size;
      static const int default_report_frequency;

      static const string delta_network_weights_id;
      static const string previous_delta_network_weights_id;
      static const string adjusted_network_weights_id;
      static const string best_weights_id;
      static const string failback_weights_id;

      /* ***********************************************
       *    Stopping criteria parameters
       */
      long max_training_epochs;
      double perf_goal;
      double min_perf_gradient;
      int max_validation_fail;

      /* ************************************************
       *    Network training mode parameters
       */
      double learning_momentum;
      bool batch_mode;
      unsigned int batch_size;
      bool random_order_mode;
      bool verbose_mode;
      double best_score;

      unsigned int critic_batch_size;
      unsigned int actor_batch_size;

      bool print_gradient;

      /* ************************************************
       *    Performance data reporting parameters
       */
      int report_freq;

   private:

      enum Prediction_Mode
      {
         FINAL_COST, CUMULATIVE_COST
      };

      /* ***********************************************
       *    Error function
       */
      SumSquaredError error_func;

      /* ***********************************************
       *    The actor-critic model
       */
      ActorCriticNet3 &actor_critic_model;

      /* ***********************************************
       *    Protected working storage data members
       */
      map<string, NetworkWeightsData> network_weights_cache;

      /* ************************************************
       *    Training performance record
       */
      TrainingRecord last_training_record;

      /* ************************************************
       *   learning rate policy
       */
      //map<string, DeltaBarDeltaLearningRate*> network_learning_rates_map;
      map<string, ConstantLearningRate *> network_learning_rates_map;

      double slow_lr_multiplier;
      double lambda;
      double gamma;
      Prediction_Mode predict_mode;

      map<string, vector<double> > prev_critic_dEdB_map;
      map<string, Array<double> > prev_critic_dEdW_map;
      map<string, Array<double> > prev_critic_Hv_map;

      map<string, Array<double> > w_critic_map;

      map<string, vector<double> > prev_actor_dEdB_map;
      map<string, Array<double> > prev_actor_dEdW_map;
      map<string, Array<double> > prev_actor_Hv_map;

      map<string, Array<double> > w_actor_map;

      map<string, Array<double>> gE_td_phi;
      double gE_norm;
   };

   inline
   void ActorCriticTrainer3::set_max_epochs (long _epochs)
   {
      max_training_epochs = _epochs;
   }

   inline
   void ActorCriticTrainer3::set_lambda (double l)
   {
      lambda = l;
   }

   inline
   void ActorCriticTrainer3::set_gamma (double g)
   {
      gamma = g;
   }

   inline
   void ActorCriticTrainer3::set_slow_learning_rate_multiplier (double _val)
   {
      slow_lr_multiplier = _val;
   }

   inline
   void ActorCriticTrainer3::set_learning_momentum (double _rate)
   {
      if (_rate < 0)
      {
         ostringstream err_str;
         err_str << "BaseTrainer::set_learning_momentum() Error : " << endl;
         err_str << "   Attempt to set learning momentum to illegal value - " << _rate << "." << endl;
         err_str << "   The value must be greater than or equal to zero." << endl;
         throw invalid_argument (err_str.str ());
      }
      learning_momentum = _rate;
   }

   inline
   void ActorCriticTrainer3::set_performance_goal (double _perf)
   {
      perf_goal = _perf;
   }

   inline
   void ActorCriticTrainer3::set_min_gradient (double _min)
   {
      min_perf_gradient = _min;
   }

   inline
   void ActorCriticTrainer3::set_max_validation_failures (int _count)
   {
      max_validation_fail = _count;
   }

   inline
   void ActorCriticTrainer3::set_batch_mode ()
   {
      batch_mode = true;
      batch_size = 0;
   }

   inline
   void ActorCriticTrainer3::set_minibatch_mode (unsigned int _size)
   {
      batch_mode = true;
      batch_size = _size;
   }

   inline
   void ActorCriticTrainer3::set_online_mode ()
   {
      batch_mode = false;
   }

   inline
   void ActorCriticTrainer3::set_predict_final_cost ()
   {
      predict_mode = FINAL_COST;
   }

   inline
   void ActorCriticTrainer3::set_predict_cumulative_cost ()
   {
      predict_mode = CUMULATIVE_COST;
   }

   inline
   void ActorCriticTrainer3::set_random_order_mode (bool _val)
   {
      random_order_mode = _val;
   }

   inline
   void ActorCriticTrainer3::set_verbose (bool _mode)
   {
      verbose_mode = _mode;
   }

   inline
   void ActorCriticTrainer3::set_print_gradient (bool _val)
   {
      print_gradient = _val;
   }

   inline
   void ActorCriticTrainer3::set_report_frequency (int _freq)
   {
      report_freq = _freq;
   }

   inline
   void ActorCriticTrainer3::set_critic_batch_size (unsigned int _sz)
   {
      critic_batch_size = _sz;
   }

   inline
   void ActorCriticTrainer3::set_actor_batch_size (unsigned int _sz)
   {
      actor_batch_size = _sz;
   }

   inline
   long ActorCriticTrainer3::max_epochs ()
   {
      return max_training_epochs;
   }

   inline
   double ActorCriticTrainer3::performance_goal ()
   {
      return perf_goal;
   }

   inline
   double ActorCriticTrainer3::min_gradient ()
   {
      return min_perf_gradient;
   }

   inline
   int ActorCriticTrainer3::max_validation_failures ()
   {
      return max_validation_fail;
   }

   inline
   bool ActorCriticTrainer3::is_batch_mode ()
   {
      return batch_mode && (batch_size == 0);
   }

   inline
   bool ActorCriticTrainer3::is_minibatch_mode ()
   {
      return batch_mode && (batch_size > 0);
   }

   inline
   bool ActorCriticTrainer3::is_online_mode ()
   {
      return !batch_mode;
   }

   inline
   unsigned int ActorCriticTrainer3::minibatch_size ()
   {
      return batch_size;
   }

   inline
   bool ActorCriticTrainer3::is_random_order_mode ()
   {
      return random_order_mode;
   }

   inline
   bool ActorCriticTrainer3::is_verbose_mode ()
   {
      return verbose_mode;
   }

   inline
   int ActorCriticTrainer3::report_frequency ()
   {
      return report_freq;
   }

   inline
   double ActorCriticTrainer3::global_perf ()
   {
      return last_training_record.best_training_perf ();
   }

   inline
   const TrainingRecord &ActorCriticTrainer3::training_record ()
   {
      return last_training_record;
   }

   inline
   void ActorCriticTrainer3::clear_training_record ()
   {
      last_training_record.clear ();
   }

   inline
   const string &ActorCriticTrainer3::genkey (const TDCNeuralNet &_net, const string &_id)
   {
      static string key;

      key = _net.name () + "_" + _id;
      return key;
   }

   inline
   void ActorCriticTrainer3::activate_actor_critic (const Pattern &_currStateVec, const Pattern &_actionVec)
   {

   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_ACTORCRITIC_TRAINER3_H_ */
