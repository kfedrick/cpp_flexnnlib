//
// Created by kfedrick on 7/29/19.
//

#ifndef FLEX_NEURALNET_TRAINERCONFIG_H_
#define FLEX_NEURALNET_TRAINERCONFIG_H_

#include <sstream>
#include <stdexcept>

namespace flexnnet
{

   class TrainerConfig
   {
   public:
      /* ***************************************************
       *    Default values
       */
      static constexpr size_t DEFAULT_TRAINING_RUNS = 1;
      static constexpr double DEFAULT_ERROR_GOAL = 0.0;
      static constexpr size_t DEFAULT_MAX_EPOCHS = 10;
      static constexpr double DEFAULT_FAILBACK_PERF_DELTA = 0.05;
      static constexpr size_t DEFAULT_MAX_VALIDATION_FAIL = 10;
      static constexpr size_t DEFAULT_BATCH_MODE = 0;
      static constexpr size_t DEFAULT_REPORT_FREQ = 1;
      static constexpr size_t DEFAULT_DISPLAY_FREQ = 1;
      static constexpr size_t DEFAULT_SAVED_NNET_LIMIT = 1;

   public:
      /* ***************************************************
       *    Public setter methods
       */

      /**
       * Set maximum training training runs. The network will be trained from
       * random initial conditions to convergence '_runs' times; _runs must be
       * greater than zero.
       *
       * @param _runs
       */
      void set_training_runs(size_t _runs);

      /**
       * Set maximum training epochs. Training will stop after at most _epochs training
       * epochs. The value of _epochs must be greater than zero.
       *
       * @param _epochs
       */
      void set_max_epochs(size_t _epochs);

      /**
       * Set performance error goal. Training will stop after performance error is less
       * than _perf. The performance error goal must be greater than or equal to zero.
       *
       * @param _perf
       */
      void set_error_goal(double _perf);

      /**
       * Set max validation failures. Training will stop after validation error increases
       * _count times.
       *
       * @param _count
       */
      void set_max_validation_failures(size_t _count);

      /**
       * Set limit for training error increase from one epoch to the next.
       * Increases beyond this limit may trigger a failback to previous weights
       * or termination of training depending on training configuration.
       *
       * @param _limit
       */
      void set_error_increase_limit(double _limit);

      /**
       * set_batch_mode(size_t _size)
       *
       * Set training batch size:
       *
       * @param _size : training batch mode/size
       *
       *    0 : full batch training, update after entire training set is presented
       *    1 : online training: update after each exemplar
       *   >1 : mini-batch size - update after _size exemplars
       *
       */
      void set_batch_mode (size_t _mode);

      /**
       * Set reporting frequency for training trace record. Training trace record will
       * report results for the first 10 entries then on every epoch thereafter where
       * (epoch mod _freq) = 0 if _freq > 0.  If _freq is zero then training data will
       * be recorded only for the final training epoch.
       *
       * @param _freq
       */
      void set_report_frequency(size_t _freq);

      /**
       * Set the frequency for displaying the training trace info during training. Training
       * trace summary results will be displayed for the first 10 entries then on every
       * epoch thereafter where (epoch mod _freq) = 0 if _freq > 0. If _freq is zero then
       * no training trace will be displayed only for the final training epoch.
       *
       * @param _freq
       */
      void set_display_frequency(size_t _freq);

      /**
       * Set the maximum number of trained neural network weights to save when
       * using multiple training runs. The trainer will save the best performing
       * networks up to _limit where _limit > 0. DEFAULT: _limit = 1
       *
       * @param _limit
       */
      void set_saved_nnet_limit(size_t _limit);


      /* ****************************************************
       *    Public getter methods
       */

      size_t training_runs(void) const;
      size_t max_epochs(void) const;
      double error_goal(void) const;
      double error_increase_limit(void) const;
      size_t batch_mode (void) const;
      size_t max_validation_failures(void) const;
      size_t report_frequency(void) const;
      size_t display_frequency(void) const;
      size_t saved_nnet_limit(void) const;

   private:

      size_t max_training_runs { DEFAULT_TRAINING_RUNS };
      size_t max_training_epochs { DEFAULT_MAX_EPOCHS };
      double min_performance_error { DEFAULT_ERROR_GOAL };
      double failback_performance_delta { DEFAULT_FAILBACK_PERF_DELTA };
      size_t training_batch_mode {DEFAULT_BATCH_MODE };
      size_t max_allowed_validation_failures { DEFAULT_MAX_VALIDATION_FAIL };
      size_t training_report_frequency { DEFAULT_REPORT_FREQ };
      size_t training_display_frequency { DEFAULT_DISPLAY_FREQ };
      size_t max_saved_nnet { DEFAULT_SAVED_NNET_LIMIT };
   };

   inline void TrainerConfig::set_training_runs(size_t _runs)
   {
      static std::ostringstream err_str;

      if (_runs == 0)
      {
         err_str.clear();
         err_str << "Error : set_max_runs - max training runs must be greater than zero\n";
         throw std::invalid_argument (err_str.str ());
      }

      max_training_runs = _runs;
   }

   inline void TrainerConfig::set_max_epochs(size_t _epochs)
   {
      static std::ostringstream err_str;

      if (_epochs == 0)
      {
         err_str.clear();
         err_str << "Error : set_max_epochs - max epochs must be greater than zero\n";
         throw std::invalid_argument (err_str.str ());
      }

      max_training_epochs = _epochs;
   }

   inline void TrainerConfig::set_error_goal(double _perf)
   {
      static std::ostringstream err_str;

      if (_perf < 0)
      {
         err_str.clear();
         err_str << "Error : set_error_goal - min error goal must be >= 0\n";
         throw std::invalid_argument (err_str.str ());
      }

      min_performance_error = _perf;
   }

   inline void TrainerConfig::set_error_increase_limit(double _limit)
   {
      static std::ostringstream err_str;

      if (_limit < 0)
      {
         err_str.clear();
         err_str << "Error : set_error_increase_limit - min value must be >= 0\n";
         throw std::invalid_argument (err_str.str ());
      }

      failback_performance_delta = _limit;
   }

   inline void TrainerConfig::set_max_validation_failures (size_t _count)
   {
      static std::ostringstream err_str;

      if (_count < 1)
      {
         err_str.clear();
         err_str << "Error : set_max_validation_failures - min count must be > 0\n";
         throw std::invalid_argument (err_str.str ());
      }

      max_allowed_validation_failures = _count;
   }

   inline void TrainerConfig::set_batch_mode (size_t _mode)
   {
      training_batch_mode = _mode;
   }

   inline void TrainerConfig::set_report_frequency(size_t _freq)
   {
      training_report_frequency = _freq;
   }

   inline void TrainerConfig::set_display_frequency(size_t _freq)
   {
      training_display_frequency = _freq;
   }

   inline void TrainerConfig::set_saved_nnet_limit (size_t _limit)
   {
      static std::ostringstream err_str;

      if (_limit < 1)
      {
         err_str.clear();
         err_str << "Error : set_max_validation_failures - min count must be > 0\n";
         throw std::invalid_argument (err_str.str ());
      }

      max_saved_nnet = _limit;
   }

   inline size_t TrainerConfig::training_runs(void) const
   {
      return max_training_runs;
   }

   inline size_t TrainerConfig::max_epochs(void) const
   {
      return max_training_epochs;
   }

   inline double TrainerConfig::error_goal(void) const
   {
      return min_performance_error;
   }

   inline double TrainerConfig::error_increase_limit(void) const
   {
      return failback_performance_delta;
   }

   inline size_t TrainerConfig::batch_mode (void) const
   {
      return training_batch_mode;
   }

   inline size_t TrainerConfig::max_validation_failures(void) const
   {
      return max_allowed_validation_failures;
   }

   inline size_t TrainerConfig::report_frequency(void) const
   {
      return training_report_frequency;
   }

   inline size_t TrainerConfig::display_frequency(void) const
   {
      return training_display_frequency;
   }

   inline size_t TrainerConfig::saved_nnet_limit(void) const
   {
      return max_saved_nnet;
   }
}

#endif //FLEX_NEURALNET_TRAINERCONFIG_H_
