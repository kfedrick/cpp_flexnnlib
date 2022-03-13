//
// Created by kfedrick on 5/29/21.
//

#ifndef FLEX_NEURALNET_ACTORCRITICDEEPRLALGO_H_
#define FLEX_NEURALNET_ACTORCRITICDEEPRLALGO_H_

#include <memory>
#include <iostream>
#include <limits>
#include <flexnnet.h>

#include <TrainingRecord.h>
#include <TrainingReport.h>
#include <BaseTrainer.h>
#include <TDTrainerConfig.h>
#include <Reinforcement.h>
#include <ActorCriticEvaluator.h>

namespace flexnnet
{

   template<class State, class Action, size_t N, template<class, class, size_t> class ACNN,
      template<class, class, size_t> class Env, template<class> class FitFunc, class LRPolicy>
   class ActorCriticDeepRLAlgo : public TrainerConfig, public TDTrainerConfig, public LRPolicy, public BaseTrainer
   {
   private:
      typedef decltype(std::tuple_cat(std::declval<State>().get_features(),
                                      std::declval<Action>().get_features())) StateActionTuple;
      typedef FeatureSetImpl<StateActionTuple> StateAction;

   public:
      ActorCriticDeepRLAlgo(ACNN<State, Action, N>& _acnnet);

      void train(Env<State, Action, N>& _env);

   protected:
      /**
       * Train the network once starting with weights initialized as
       * specified by the current policy.
       *
       * @param _trnset
       */
      TrainingRecord train_one_run(Env<State, Action, N>& _env);

      void train_episode(size_t _epoch, Env<State, Action, N>& _env);

      virtual void calc_weight_updates(const ValarrMap _extern_r);
      virtual void calc_actor_weight_updates(const ValarrMap _extern_r);
      virtual void calc_critic_weight_updates(const ValarrMap _extern_r);

      double update_performance_traces(unsigned int _epoch, double _trnperf, TrainingRecord& _trec) {};

      void failback();
      bool failback_test(double _trnperf, double _prev_trnperf);

      void zero_actor_eligibility_traces();

      void update_actor_eligibility_traces();

      void zero_critic_eligibility_traces();

      void update_critic_eligibility_traces();

      bool actor_training_epoch(unsigned int _epoch) const;

   private:
      void alloc();

   private:
      ACNN<State, Action, N>& nnet;

      FitFunc<Action> fitnessfunc;
      ActorCriticEvaluator<State, Action, N, ACNN, Env, FitFunc> evaluator;

      Env<State, Action, N> validation_env;
      Env<State, Action, N> test_env;

      // working storage to calculate weight updates
      std::map<std::string, Array2D<double>> actor_weight_updates;
      std::map<std::string, Array2D<double>> critic_weight_updates;

      LRPolicy actor_learning_rates;
      LRPolicy critic_learning_rates;

      std::map<std::string, Array2D<double>> actor_eligibility_trace;
      std::map<std::string, Array2D<double>> critic_eligibility_trace;

      TrainingRecord training_record;
   };

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::ActorCriticDeepRLAlgo(
      AC<S, A, N>& _nnet) : nnet(_nnet), actor_learning_rates(_nnet.get_actor()), critic_learning_rates(_nnet.get_critic())
   {
      alloc();
      evaluator.set_sampling_count(500);
      critic_learning_rates.set_learning_rate(0.0005);

      const NeuralNet <S, A>& actor = nnet.get_actor();
      const std::map<std::string, std::shared_ptr<NetworkLayer>>& actor_layers = actor.get_layers();
      for (auto it : actor_layers)
      {
         std::string id = "actor:" + it.first;

         // Set to train layer biases by default.
         TrainerConfig::set_train_biases(id, true);
      }

      const NeuralNet<StateAction , Reinforcement<N>>& critic = nnet.get_critic();
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & critic_layers = critic.get_layers();
      for (auto it : critic_layers)
      {
         std::string id = "critic:" + it.first;

         // Set to train layer biases by default.
         TrainerConfig::set_train_biases(id, true);
      }
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   void
   ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::alloc()
   {
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & actor_layers = nnet.get_actor().get_layers();

      actor_weight_updates.clear();
      for (auto layer : actor_layers)
      {
         std::string id = layer.first;
         const LayerWeights& w = layer.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         actor_eligibility_trace[layer.first].set(layer.second->dEdw());

         actor_weight_updates[id] = {};
         actor_weight_updates[id].resize(dim.rows, dim.cols);
      }

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & critic_layers = nnet.get_critic().get_layers();

      critic_weight_updates.clear();
      for (auto layer : critic_layers)
      {
         std::string id = layer.first;
         const LayerWeights& w = layer.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         //std::cout << "setting critic eligibility trace " << id << "\n" << std::flush;
         critic_eligibility_trace[layer.first].set(layer.second->dEdw());

         critic_weight_updates[id] = {};
         critic_weight_updates[id].resize(dim.rows, dim.cols);
      }
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::train(
      Env<S, A, N>& _env)
   {
      std::cout << "ActorCriticDeepRLAlgo.train()\n" << std::flush;

      double trn_perf, trn_stdev;
      double perf;

      alloc();

      size_t no_runs = TrainerConfig::training_runs();
      for (size_t runndx = 0; runndx < no_runs; runndx++)
      {
         training_record.clear();

         if (runndx > 0)
         {
            //nnet.get_actor().initialize_weights();
            //nnet.get_critic().initialize_weights();
         }

         save_network_weights(nnet.get_actor(), "initial_weights");
         save_network_weights(nnet.get_critic(), "initial_weights");

         /*
          * Evaluate and save the performance for the initial network
          */
         std::tie(trn_perf, trn_stdev) = evaluator.evaluate(this->nnet, _env);
         perf = update_performance_traces(0, trn_perf, training_record);
         std::cout << trn_perf << "\n" << std::flush;

         training_record.best_epoch = 0;
         training_record.best_performance = perf;
         save_network_weights(nnet.get_actor(), "best_epoch");
         save_network_weights(nnet.get_critic(), "best_epoch");

         // *** train the network
         train_one_run(_env);

/*         const std::map<std::string, std::shared_ptr<NetworkLayer>>
            & layers = this->nnet.get_layers();
         for (auto& it : layers)
            training_record.network_weights[it.first] = it.second->weights();*/

         save_training_record(training_record);

         // TODO - update aggregate training statistics
      }
      std::cout << "ActorCriticDeepRLAlgo.train() EXIT\n" << std::flush;
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
         inline
   TrainingRecord ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::train_one_run(Env<S, A, N>& _env)
   {
      std::cout << "ActorCriticDeepRLAlgo.train_one_run()\n" << std::flush;

      double trn_perf, trn_stdev;
      double trn_perf_degradation;
      double perf;

      unsigned int consecutive_failback_count = 0;

      actor_learning_rates.reset();
      critic_learning_rates.reset();

      // Previous performance vectorize - used for failback testing
      double prev_trn_perf = std::numeric_limits<double>::max();
      double failback_limit = TrainerConfig::error_increase_limit();

      // Init best performance assuming we are trying to minimize error
      training_record.stop_signal = TrainingStopSignal::UNKNOWN;

      // Iterate through training epochs
      bool pending_updates = false;
      size_t n_epochs = TrainerConfig::max_epochs();
      size_t epoch = 0;
      for (epoch = 0; epoch < n_epochs; epoch++)
      {
         //std::cout << "train epoch " << epoch << "\n" << std::flush;

         // Save the network weights in case we need to fail back
         save_network_weights(nnet.get_actor(), "failback");
         save_network_weights(nnet.get_critic(), "failback");

         // Call function to iterate over training samples and update
         // the network weights.
         train_episode(epoch, _env);
         pending_updates = true;

         // If training in online or mini-batch mode, update now.
         if (TrainerConfig::batch_mode() > 0
             && epoch % TrainerConfig::batch_mode() == 0)
         {
            adjust_network_weights(nnet.get_critic());
            pending_updates = false;
         }

         // Evaluate the performance of the updated network
         prev_trn_perf = trn_perf;
         std::tie(trn_perf, trn_stdev) = evaluator.evaluate(nnet, _env);

         //std::cout << "trn error : " << trn_perf << "\n" << std::flush;
         std::cout << trn_perf << "\n" << std::flush;

         /*
          * If the performance on the training set worsens by an
          * amount greater than the fail-back limit then (1) restore
          * the previous weights, (2) lower the learning rates and
          * retry the epoch.
          */
         if (failback_test(trn_perf, prev_trn_perf))
         {
/*            failback();
            epoch--;

            consecutive_failback_count++;
            if (consecutive_failback_count > this->max_failbacks())
            {
               training_record.stop_signal = TrainingStopSignal::MAX_FAILBACK_REACHED;
               return training_record;
            }
            continue;*/
         }
         else
         {
            consecutive_failback_count = 0;
            actor_learning_rates.apply_learning_rate_adjustments();
            critic_learning_rates.apply_learning_rate_adjustments();
         }

         // Update performance history in training record
         if (epoch < 10 || epoch % TrainerConfig::report_frequency() == 0
             || epoch == n_epochs - 1)
            perf = update_performance_traces(epoch + 1, trn_perf, training_record);

         // Call function to save network weights for the best epoch.
         if (perf < training_record.best_performance)
         {
            training_record.best_epoch = epoch + 1;
            training_record.best_performance = perf;

            save_network_weights(nnet.get_actor(), "best_epoch");
            save_network_weights(nnet.get_critic(), "best_epoch");
         }

         // If we've reached the target error goal then exit.
         if (perf < TrainerConfig::error_goal())
         {
            training_record.stop_signal = TrainingStopSignal::CRITERIA_MET;
            break;
         }
      }

      // If training in batch mode or there are updates pending from
      // an mini-batch then update weights now
      if (pending_updates)
         std::cout << "pending updates = " << pending_updates << "\n";

      if (TrainerConfig::batch_mode() == 0 || pending_updates)
      {
         adjust_network_weights(nnet.get_critic());

         prev_trn_perf = trn_perf;
         std::tie(trn_perf, trn_stdev) = evaluator.evaluate(nnet, _env);

         std::cout << "final trn error : " << trn_perf << "\n" << std::flush;

         // training failback???
         if (failback_test(trn_perf, prev_trn_perf))
            failback();

         perf = update_performance_traces(epoch + 1, trn_perf, training_record);

         // Call function to save network weights for the best epoch.
         if (perf < training_record.best_performance)
         {
            training_record.best_epoch = epoch + 1;
            training_record.best_performance = perf;

            save_network_weights(nnet.get_actor(), "best_epoch");
            save_network_weights(nnet.get_critic(), "best_epoch");
         }

         // If we've reached the target error goal then exit.
         if (perf < TrainerConfig::error_goal())
            training_record.stop_signal = TrainingStopSignal::CRITERIA_MET;
      }

      if (training_record.stop_signal == TrainingStopSignal::UNKNOWN)
         training_record.stop_signal = TrainingStopSignal::MAX_EPOCHS_REACHED;

      // Restore the best network weights.
      restore_network_weights(nnet.get_actor(), "best_epoch");
      restore_network_weights(nnet.get_critic(), "best_epoch");

      //std::cout << "SupervisedTrainingAlgo.train_one_run() EXIT\n" << std::flush;
      return training_record;
   }



   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::train_episode(size_t _epoch, Env<S,A,N>& _env)
   {
      //std::cout << "ActorCriticDeepRLAlgo.train_episode\n" << std::flush;

      Reinforcement<N> previous_V_est;
      Reinforcement<N> V_est;

      // Negative 1.0 array for calculating critic dydw
      ValarrMap ones = nnet.get_critic().value_map();
      for (auto& it : ones)
         it.second = -1.0;

      Reinforcement<N> zero_V_est("R");
      zero_V_est.decode({{0.0}});

      Reinforcement<N> V_tgt("R");
      V_tgt.decode({{1.0}});

      /*
       * Present the first item in the series and calculate the
       * initial eligibility trace info
       */
      const S& state = _env.reset();

/*      std::cout << "init State: ";
      const S& istate = _env.state();
      std::valarray<double> iv = std::get<0>(istate.get_features()).get_encoding();
      for (int i=0; i<iv.size(); i++)
      {
         std::cout << iv[i] << " ";
      }
      std::cout << "\n";*/

      const std::tuple<A,Reinforcement<N>>& nnout = nnet.activate(state);
      V_est = std::get<1>(nnout);

      zero_critic_eligibility_traces();
      //std::cout << "ones(" << ones.begin()->first << ") " << ones.begin()->second[0] << "\n";

      nnet.backprop_critic(ones);
      update_critic_eligibility_traces();

      _env.next(std::get<0>(nnout).get_action());

      int series_ndx = 1;
      while (!_env.is_terminal())
      {
         // V(t)
         const Reinforcement<N>& R = _env.get_reinforcement();

         const S& state = _env.state();

/*         std::cout << "State (" << series_ndx++ << ") : ";
         std::valarray<double> v = std::get<0>(state.get_features()).get_encoding();
         for (int i=0; i<v.size(); i++)
         {
            std::cout << v[i] << " ";
         }
         std::cout << "\n";*/

         // V estimate (t)
         const std::tuple<A,Reinforcement<N>> nnout = nnet.activate(state);
         previous_V_est = V_est;
         V_est = std::get<1>(nnout);

         //std::cout << "R, V_est : " << R[0] << ", " << V_est[0] << " " << previous_V_est[0] << "\n" << std::flush;

         //const ValarrMap& td_gradient = evaluator.calc_td_error_gradient(R, V_est, previous_V_est);
         const ValarrMap& td_gradient = evaluator.calc_td_error_gradient(zero_V_est, V_est, previous_V_est);
         calc_critic_weight_updates(td_gradient);

         if (actor_training_epoch(_epoch))
         {
            const ValarrMap& actor_gradient = evaluator.calc_actor_error_gradient(V_tgt, previous_V_est);
            calc_actor_weight_updates(actor_gradient);
         }

         // Update the eligibility traces
         nnet.backprop_critic(ones);
         update_critic_eligibility_traces();

         _env.next(std::get<0>(nnout).get_action());
      }

      // print state
/*      std::cout << "final State: ";
      const S& state2 = _env.state();
      std::valarray<double> v = std::get<0>(state2.get_features()).get_encoding();
      for (int i=0; i<v.size(); i++)
      {
         std::cout << v[i] << " ";
      }
      std::cout << "\n";*/

      const Reinforcement<N>& R = _env.get_reinforcement();
      //std::cout << "final (R,Vest) = " << R[0] << "," << V_est[0] << "\n" << std::flush;
      const ValarrMap& td_gradient = evaluator.calc_td_error_gradient(R, zero_V_est, V_est);
      calc_critic_weight_updates(td_gradient);

      if (actor_training_epoch(_epoch))
      {
         const ValarrMap& actor_gradient = evaluator.calc_actor_error_gradient(V_tgt, V_est);
         calc_actor_weight_updates(actor_gradient);
      }
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void
   ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::calc_weight_updates(const ValarrMap _tdgradient)
   {
      //std::cout << "calc_weight_updates(" << _tderr << ")\n" << std::flush;
      double _tderr = _tdgradient.begin()->second[0];

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & layers = nnet.get_critic().get_layers();
      for (auto& it : layers)
      {
         std::string id = it.first;
         Array2D<double> lr = critic_learning_rates.get_learning_rates(id);
         std::cout << "critic learning rate = " << lr.at(0, 0) << "\n";

         const Array2D<double> etrace_dEdw = critic_eligibility_trace.at(id);

         const Array2D<double>::Dimensions dims = critic_weight_updates[id].size();

         // If this layer doesn't train biases, stop before the last column
         unsigned int
            last_col = (TrainerConfig::train_biases("critic:"+id)) ? dims.cols : dims.cols - 1;
         critic_weight_updates[id] = 0;
         for (unsigned int row = 0; row < dims.rows; row++)
            for (unsigned int col = 0; col < last_col; col++)
               critic_weight_updates[id].at(row, col) = -lr.at(row, col) * _tderr * etrace_dEdw.at(row, col);

         accumulate_weight_updates(nnet.get_critic(), id, critic_weight_updates[id]);
      }
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void
   ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::calc_critic_weight_updates(const ValarrMap _tdgradient)
   {
      //std::cout << "size of tdgradient " << _tdgradient.size() << "\n";
      double _tderr = _tdgradient.begin()->second[0];
      //std::cout << "calc_critic_weight_updates(" << _tderr << ")\n" << std::flush;

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & layers = nnet.get_critic().get_layers();
      for (auto& it : layers)
      {
         std::string id = it.first;

         Array2D<double> lr = critic_learning_rates.get_learning_rates(id);

         const Array2D<double> etrace_dEdw = critic_eligibility_trace.at(id);

         const Array2D<double>::Dimensions dims = critic_weight_updates[id].size();

         // If this layer doesn't train biases, stop before the last column
         unsigned int
            last_col = (TrainerConfig::train_biases("critic:"+id)) ? dims.cols : dims.cols - 1;

         critic_weight_updates[id] = 0;
         for (unsigned int row = 0; row < dims.rows; row++)
            for (unsigned int col = 0; col < last_col; col++)
            {
               critic_weight_updates[id].at(row, col) =
                  -lr.at(row, col) * _tderr * etrace_dEdw.at(row, col);
               //std::cout << etrace_dEdw.at(row, col) << " " << critic_weight_updates[id].at(row, col) << "\n";
            }
         accumulate_weight_updates(nnet.get_critic(), id, critic_weight_updates[id]);
      }
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void
   ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::calc_actor_weight_updates(const ValarrMap _actor_gradient)
   {
      //std::cout << "calc_actor_weight_updates(" << _actor_gradient.begin()->second[0] << ")\n" << std::flush;

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & layers = this->nnet.get_actor().get_layers();
      for (auto it : layers)
      {
         std::string id = it.first;

         Array2D<double> lr = actor_learning_rates.get_learning_rates(id);

         const Array2D<double> dE_dw = it.second->dEdw();

         const Array2D<double>::Dimensions dims = actor_weight_updates[id].size();

         // If this layer doesn't train biases, stop before the last column
         unsigned int
            last_col = (TrainerConfig::train_biases("actor:"+id)) ? dims.cols : dims.cols - 1;

         actor_weight_updates[id] = 0;
         for (unsigned int row = 0; row < dims.rows; row++)
            for (unsigned int col = 0; col < last_col; col++)
               actor_weight_updates[id].at(row, col) = -lr.at(row, col) * dE_dw.at(row, col);

         accumulate_weight_updates(nnet.get_actor(), id, actor_weight_updates[id]);
      }
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::failback()
   {
      std::cout << "failback\n";

      restore_network_weights(nnet.get_actor(), "failback");
      restore_network_weights(nnet.get_critic(), "failback");

      actor_learning_rates.reduce_learning_rate();
      critic_learning_rates.reduce_learning_rate();
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   bool ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::failback_test(double _trnperf, double _prev_trnperf)
   {
      /*
       * If the performance on the training set worsens by an
       * amount greater than the fail-back limit then (1) restore
       * the previous weights, (2) lower the learning rates and
       * retry the epoch.
       */
      double trn_perf_degradation =
         (_prev_trnperf > 0) ? (_trnperf - _prev_trnperf) / _prev_trnperf :
         (_trnperf - 1e-9) / 1e-9;
      if (trn_perf_degradation > TrainerConfig::error_increase_limit())
         return true;
      else
         return false;
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   bool ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::actor_training_epoch(unsigned int _epoch) const
   {
      if (_epoch > 5 && _epoch%2 == 0)
         return true;
      else
         return false;
   }

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void
   ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::zero_actor_eligibility_traces()
   {

   };

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void
   ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::update_actor_eligibility_traces()
   {

   };

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void
   ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::zero_critic_eligibility_traces()
   {
      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = this->nnet.get_critic().get_layers();
      for (auto& layer : network_layers)
         critic_eligibility_trace[layer.first] = 0;
   };

   template<class S, class A, size_t N, template<class, class, size_t> class AC,
      template<class, class, size_t> class Env, template<class> class Fit, class LR>
   inline
   void
   ActorCriticDeepRLAlgo<S, A, N, AC, Env, Fit, LR>::update_critic_eligibility_traces()
   {
      //std::cout << "update_critic_eligibility_traces()\n" << std::flush;

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = nnet.get_critic().get_layers();
      for (auto& layer : network_layers)
      {
         //std::cout << "dEdw = " << layer.first << " " << layer.second->dEdw().at(0,0) << "\n" << std::flush;
         critic_eligibility_trace[layer.first] =
            layer.second->dEdw() + get_lambda() * critic_eligibility_trace[layer.first];
      }
   };

} // end namespace flexnnet

#endif // FLEX_NEURALNET_ACTORCRITICDEEPRLALGO_H_
