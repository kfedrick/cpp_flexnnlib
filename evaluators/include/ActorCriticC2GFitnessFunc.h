//
// Created by kfedrick on 3/25/22.
//

#ifndef FLEX_NEURALNET_ACTORCRITICC2GFITNESSFUNC_H_
#define FLEX_NEURALNET_ACTORCRITICC2GFITNESSFUNC_H_

#include <Environment.h>

using flexnnet::Environment;

namespace flexnnet
{

   template<class State,
      class Action, unsigned int RSZ, template<class, class, unsigned int>
      class Env> class ActorCriticC2GFitnessFunc
   {
      using NNTyp = BaseActorCriticNetwork<State, Action, RSZ>;
      using EnvTyp = Env<State, Action, RSZ>;
      using RTyp = Reinforcement<RSZ>;

   public:
      static constexpr double DEFAULT_SUBSAMPLE_SZ = 3;

   public:
      void set_subsample_count(unsigned int _sz);

      virtual std::tuple<double, double> calc_fitness_standard_error(
         NNTyp& _nnet, EnvTyp& _tstset, unsigned int _subsample_sz = DEFAULT_SUBSAMPLE_SZ);

      double calc_fitness(
         NNTyp& _nnet, EnvTyp& _env, unsigned int _subsample_sz = DEFAULT_SUBSAMPLE_SZ);

      double evaluate_sample(
         NNTyp& _nnet, EnvTyp& _env);

      double calc_tde_gradient(
         const RTyp& _R_t1, const RTyp& _Re_t1, const RTyp& _Re_t0, ValarrMap& _err, double _E = 1);

   private:
      std::vector<double> tde_cache;
      double min_r, max_r, range_r;

      std::valarray<double> performance_cache;
   };

   template<class State,
      class Action, unsigned int RSZ, template<class, class, unsigned int>
      class Env>
   void ActorCriticC2GFitnessFunc<State, Action, RSZ, Env>::set_subsample_count(unsigned int _sz)
   {
      if (_sz == 0)
         throw std::invalid_argument("invalid subsample size : 0 (zero)");

      performance_cache.resize(_sz);
   }

   template<class State,
      class Action, unsigned int RSZ, template<class, class, unsigned int>
      class Env> double ActorCriticC2GFitnessFunc<State, Action, RSZ, Env>::calc_fitness(
      NNTyp& _nnet, EnvTyp& _env, unsigned int _subsample_sz)
   {
      double series_sumsqr_tde;
      double dataset_sumsqr_tde = 0;

      tde_cache.clear();
      min_r = std::numeric_limits<double>::max();
      max_r = 0;

      size_t sample_no = 0, step_count = 0;
      for (unsigned int sample_no = 0; sample_no < _subsample_sz; sample_no++)
      {
         _env.reset();
         evaluate_sample(_nnet, _env);
      }

      range_r = max_r - min_r;
      dataset_sumsqr_tde = 0;
      for (double tde: tde_cache)
         dataset_sumsqr_tde += (tde / range_r) * (tde / range_r);
      return (tde_cache.size() > 0) ? (sqrt(dataset_sumsqr_tde) / tde_cache.size()) : 0;
   }

   template<class State, class Action, unsigned int RSZ, template <class, class, unsigned int> class Env>
   double ActorCriticC2GFitnessFunc<State, Action, RSZ, Env>::evaluate_sample(
      NNTyp& _nnet, EnvTyp& _env)
   {
      double sum_tde;
      std::tuple<Action, Reinforcement<RSZ>> nnout0, nnout1;
      Reinforcement<RSZ> R;
      Reinforcement<RSZ> Re_t1, Re_t0;
      ValarrMap tde_gradient = _env.get_reinforcement().value_map();

      /*
       * Sum of the squared temporal difference error across all steps
       * of all series in the sub-sampling.
       */
      double series_sumsqr_tde = 0;

      const State& state = _env.state();
      nnout1 = _nnet.activate(state);
      _env.next(std::get<0>(nnout0).get_action());

      while (!_env.is_terminal())
      {
         const Reinforcement<RSZ>& R1 = _env.get_reinforcement();

         nnout0 = nnout1;
         Re_t0 = std::get<1>(nnout0);

         const State& state = _env.state();
         nnout1 = _nnet.activate(state);
         Re_t1 = std::get<1>(nnout1);

         sum_tde = calc_tde_gradient(R1, Re_t1, Re_t0, tde_gradient);
         tde_cache.push_back(sum_tde);
         series_sumsqr_tde += sum_tde * sum_tde;

         _env.next(std::get<0>(nnout1).get_action());
      }
      const Reinforcement<RSZ>& last_R = _env.get_reinforcement();

      // Assign last examplar to zeronnout just to get the feature names
      RTyp zeronnout = last_R;
      ValarrMap vm({{"R", {0}}});
      std::valarray<double> va = {0};
      zeronnout[0].decode(va);

      nnout0 = _nnet.value();
      Re_t0 = std::get<1>(nnout0);

      sum_tde = calc_tde_gradient(last_R, zeronnout, Re_t0, tde_gradient);
      tde_cache.push_back(sum_tde);

      return series_sumsqr_tde;
   }

   template<class State,
      class Action, unsigned int RSZ, template<class, class, unsigned int>
      class Env> double ActorCriticC2GFitnessFunc<State, Action, RSZ, Env>::calc_tde_gradient(
      const RTyp& _R_t1, const RTyp& _Re_t1, const RTyp& _Re_t0, ValarrMap& _err, double _E)
   {
      std::valarray<double> sqrdiff;

      const ValarrMap& R_vamap = _R_t1.value_map();
      const ValarrMap& Re_t1_vamap = _Re_t1.value_map();
      const ValarrMap& Re_t0_vamap = _Re_t0.value_map();

      // Sum of the temporal difference error across all reinforcement signals
      double sum_tde = 0;

      for (const auto& R: R_vamap)
      {
         const std::string id = R.first;
         _err[id] = R.second + 0.98 * Re_t1_vamap.at(id) - Re_t0_vamap.at(id);

         if (Re_t0_vamap.at(id).max() > max_r)
            max_r = Re_t0_vamap.at(id).max();

         if (Re_t0_vamap.at(id).min() < min_r)
            min_r = Re_t0_vamap.at(id).min();

         sum_tde += _err[id].sum();
      }

      return sum_tde;
   }

   template<class State,
      class Action, unsigned int RSZ, template<class, class, unsigned int>
      class Env> std::tuple<double, double> ActorCriticC2GFitnessFunc<State,
                                                                      Action,
                                                                      RSZ,
                                                                      Env>::calc_fitness_standard_error(
      NNTyp& _nnet, EnvTyp& _env, unsigned int _subsample_sz)
   {
      /*
       * Validate _subsample_fraction value is greater than or equal to
       * 0.0 and less than or equal to 1.0
       */
      if (_subsample_sz <= 0)
      {
         static std::stringstream sout;
         sout << "Error : LossFunction::calc_fitness_standard_error - "
              << "Invalid sub-sample fraction specified : \"" << _subsample_sz
              << "\" : must be 0 to 1.0\n";
         throw std::invalid_argument(sout.str());
      }

      performance_cache.resize(_subsample_sz);


      // Iterate through all exemplars in the training set_weights
      for (size_t sample_ndx = 0; sample_ndx < _subsample_sz; sample_ndx++)
      {
         performance_cache[sample_ndx] = calc_fitness(_nnet, _env, _subsample_sz);
      }

      // Calculate fitness mean and standard error across samples
      double sample_mean, sample_std_error;

      sample_mean = performance_cache.sum() / _subsample_sz;
      double diff, sumsqrdiff = 0;
      for (size_t i = 0; i < _subsample_sz; i++)
      {
         diff = sample_mean - performance_cache[i];
         sumsqrdiff += diff * diff;
      }
      double sample_var =
         (_subsample_sz > 30) ? sumsqrdiff / (_subsample_sz - 1) : sumsqrdiff / _subsample_sz;
      sample_std_error = sqrt(sample_var);

      return std::make_tuple(sample_mean, sample_std_error);
   }
}

#endif // FLEX_NEURALNET_ACTORCRITICC2GFITNESSFUNC_H_
