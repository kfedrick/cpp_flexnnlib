//
// Created by kfedrick on 3/27/21.
//

#ifndef FLEX_NEURALNET_TRAININGREPORT_H_
#define FLEX_NEURALNET_TRAININGREPORT_H_

#include <set>
#include <algorithm>
#include <iostream>
#include "TrainingRecord.h"

namespace flexnnet
{
   class TrainingReport
   {
   public:
      void clear(void);
      void set_max_records(unsigned int _max);
      void add_record(const TrainingRecord& _rec);

      int total_training_runs(void) const;
      std::tuple<double,double> performance_statistics(void) const;
      double successful_training_rate(void) const;
      const std::set<TrainingRecord>& get_records(void) const;
      unsigned int successful_training_runs(void) const;

   private:
      void calc_performance_statistics(void) const;

   private:
      std::set<TrainingRecord> records;

      static const unsigned int DEFAULT_MAX_RECORDS{10};

      unsigned int max_records{DEFAULT_MAX_RECORDS};

      unsigned int training_runs{0};

      // The number of runs reaching the minimum error criteria
      unsigned int successful_runs{0};

      // Working cache to hold performance for added records
      std::vector<double> temp_performance;
      double sum_performance{0};

      bool stale{false};
      mutable double mean_performance{0};
      mutable double stdev_performance{0};
   };

   inline
   void TrainingReport::clear(void)
   {
      temp_performance.clear();
      records.clear();
      training_runs = 0;
      successful_runs = 0;
      sum_performance = 0;
      stale = false;
   }

   inline
   void TrainingReport::set_max_records(unsigned int _max)
   {
      if (_max == 0)
         throw std::invalid_argument("Error : max records must be greater than zero.");

      max_records = _max;
      while (records.size() > max_records)
      {
         std::set<TrainingRecord>::iterator it = --records.end();
         records.erase(it);
      }
   }

   int TrainingReport::total_training_runs(void) const
   {
      return training_runs;
   }

   inline
   std::tuple<double,double> TrainingReport::performance_statistics(void) const
   {
      calc_performance_statistics();
      return std::tuple<double,double>(mean_performance, stdev_performance);
   }

   inline
   double TrainingReport::successful_training_rate(void) const
   {
      return (training_runs > 0) ? (double) successful_runs / training_runs : 0;
   }

   inline
   unsigned int TrainingReport::successful_training_runs(void) const
   {
      return successful_runs;
   }

   inline
   const std::set<TrainingRecord>& TrainingReport::get_records(void) const
   {
      return records;
   }

   inline
   void TrainingReport::add_record(const TrainingRecord& _rec)
   {
      // Increment training run count
      training_runs++;

      // Increment successful runs if error criteria was met
      if (_rec.stop_signal == TrainingStopSignal::CRITERIA_MET)
         successful_runs++;

      // Accumulate statistics
      temp_performance.push_back(_rec.best_performance);
      sum_performance += _rec.best_performance;

      // Insert the new record and truncate the last record
      // to reduce the set of records to max size. This will
      // remove the record with the worst performance.
      records.insert(_rec);
      if (records.size() > max_records)
      {
         std::set<TrainingRecord>::iterator it = --records.end();
         records.erase(it);
      }

      // Mark results as stale
      stale = true;
   }

   inline
   void TrainingReport::calc_performance_statistics(void) const
   {
      // Only do the computation of the results are stale
      if (!stale)
         return;

      mean_performance = 0;
      stdev_performance = 0;

      // Calculate mean performance of all training run records
      if (training_runs > 0)
      {
         mean_performance = sum_performance / training_runs;

         double variance = 0;
         for (auto& perf : temp_performance)
            variance = (mean_performance - perf) * (mean_performance - perf);
         variance /= training_runs;
         stdev_performance = sqrt(variance);
      }
   }
}

#endif //FLEX_NEURALNET_TRAININGREPORT_H_
