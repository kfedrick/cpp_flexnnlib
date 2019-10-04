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
      enum VerboseLevel {OFF, INFO, DEBUGGING};

   public:
      /* ***************************************************
       *    Default values
       */
      static constexpr VerboseLevel DEFAULT_VERBOSE = OFF;
      static constexpr double DEFAULT_ERROR_GOAL = 0.0;
      static constexpr size_t DEFAULT_MAX_EPOCHS = 10;
      static constexpr size_t DEFAULT_MAX_VALIDATION_FAIL = 10;
      static constexpr size_t DEFAULT_BATCH_MODE = 1;
      static constexpr size_t DEFAULT_REPORT_FREQ = 1;
      static constexpr size_t DEFAULT_DISPLAY_FREQ = 1;

   public:
      /* ***************************************************
       *    Constructors
       */

      TrainerConfig();

      /* ***************************************************
       *    Public setter methods
       */

      /**
       * Set maximum training epochs. Training will stop after at most _epochs training
       * epochs. The value of _epochs must be greater than zero.
       *
       * @param _epochs
       */
      void set_max_epochs(unsigned int _epochs);

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
      void set_max_validation_failures(unsigned int _count);

      /**
       * set_batch_mode(unsigned int _size)
       *
       * Set training batch size:
       *
       * @param _size : training batch size
       *
       *    0 : full batch training, update after entire training set is presented
       *    1 : online training: update after each exemplar
       *   >1 : mini-batch size - update after _size exemplars
       *
       */
      void set_batch_size (unsigned int _size);

      /**
       * Set reporting frequency for training trace record. Training trace record will
       * report results for the first 10 entries then on every epoch thereafter where
       * (epoch mod _freq) = 0 if _freq > 0.  If _freq is zero then training data will
       * be recorded only for the final training epoch.
       *
       * @param _freq
       */
      void set_report_frequency(unsigned int _freq);

      /**
       * Set the frequency for displaying the training trace info during training. Training
       * trace summary results will be displayed for the first 10 entries then on every
       * epoch thereafter where (epoch mod _freq) = 0 if _freq > 0. If _freq is zero then
       * no training trace will be displayed only for the final training epoch.
       *
       * @param _freq
       */
      void set_display_frequency(unsigned int _freq);

      /**
       * Set reporting level;
       * @param _mode
       */
      void set_verbose(VerboseLevel _level);


      /* ****************************************************
       *    Public getter methods
       */

      unsigned int get_max_epochs(void) const;
      double error_goal(void) const;
      unsigned int batch_size (void) const;
      unsigned int max_validation_failures(void) const;
      unsigned int report_frequency(void) const;
      unsigned int display_frequency(void) const;
      VerboseLevel verbose_level(void) const;

   private:
      unsigned int max_training_epochs { DEFAULT_MAX_EPOCHS };
      double min_performance_error { DEFAULT_ERROR_GOAL };
      unsigned int training_batch_size { DEFAULT_BATCH_MODE };
      unsigned int max_allowed_validation_failures { DEFAULT_MAX_VALIDATION_FAIL };
      unsigned int training_report_frequency { DEFAULT_REPORT_FREQ };
      unsigned int training_display_frequency { DEFAULT_REPORT_FREQ };
      VerboseLevel verbose { DEFAULT_VERBOSE };
   };

   inline void TrainerConfig::set_max_epochs(unsigned int _epochs)
   {
      static std::ostringstream err_str;

      if (_epochs == 0)
      {
         err_str.clear();
         err_str << "Error : set_max_epochs - max epochs must be greater than 0\n";
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

   inline void TrainerConfig::set_max_validation_failures (unsigned int _count)
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

   inline void TrainerConfig::set_batch_size (unsigned int _size)
   {
      training_batch_size = _size;
   }

   inline void TrainerConfig::set_report_frequency(unsigned int _freq)
   {
      training_report_frequency = _freq;
   }

   inline void TrainerConfig::set_display_frequency(unsigned int _freq)
   {
      training_display_frequency = _freq;
   }

   inline void TrainerConfig::set_verbose(VerboseLevel _level)
   {
      verbose = _level;
   }


   inline unsigned int TrainerConfig::get_max_epochs(void) const
   {
      return max_training_epochs;
   }

   inline double TrainerConfig::error_goal(void) const
   {
      return min_performance_error;
   }

   inline unsigned int TrainerConfig::batch_size (void) const
   {
      return training_batch_size;
   }

   inline unsigned int TrainerConfig::max_validation_failures(void) const
   {
      return max_allowed_validation_failures;
   }

   inline unsigned int TrainerConfig::report_frequency(void) const
   {
      return training_report_frequency;
   }

   inline unsigned int TrainerConfig::display_frequency(void) const
   {
      return training_display_frequency;
   }

   inline TrainerConfig::VerboseLevel TrainerConfig::verbose_level(void) const
   {
      return verbose;
   }

}

#endif //FLEX_NEURALNET_TRAINERCONFIG_H_
