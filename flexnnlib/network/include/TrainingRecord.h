/*
 * TrainingRecord.h
 *
 *  Created on: Mar 28, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_TRAININGRECORD_H_
#define FLEX_NEURALNET_TRAININGRECORD_H_

#include <vector>
#include <map>
#include <string>
#include <cfloat>
#include <cmath>

using namespace std;

namespace flex_neuralnet
{

class TrainingRecord
{
public:

   TrainingRecord()
   {
      clear();
   }

   int size() const
   {
      return training_record.size();
   }

   unsigned int training_epochs() const
   {
      return no_training_epochs;
   }

   unsigned int best_training_epoch() const
   {
      return the_best_train_epoch;
   }

   double best_training_perf() const
   {
      return the_best_train_perf;
   }

   unsigned int best_validation_epoch() const
   {
      return the_best_valid_epoch;
   }

   double best_validation_perf() const
   {
      return the_best_valid_perf;
   }

   unsigned int stop_signal() const
   {
      return the_stop_signal;
   }

   class Entry;

   Entry& at(unsigned int _ndx)
   {
      return training_record.at(_ndx);
   }

   const Entry& at(unsigned int _ndx) const
   {
      return training_record.at(_ndx);
   }

   Entry& operator[](unsigned int _ndx)
   {
      return training_record[_ndx];
   }

   const Entry& operator[](unsigned int _ndx) const
   {
      return training_record[_ndx];
   }

   void set_training_epochs(unsigned int _epochs)
   {
      no_training_epochs = _epochs;
   }

   void set_best_training_epoch(unsigned int _epoch, double _perf)
   {
      the_best_train_epoch = _epoch;
      the_best_train_perf = _perf;
   }

   void set_best_validation_epoch(unsigned int _epoch, double _perf)
   {
      the_best_valid_epoch = _epoch;
      the_best_valid_perf = _perf;
   }

   void set_stop_signal(unsigned int _sig)
   {
      the_stop_signal = _sig;
   }

   void push_back(const Entry& _entry)
   {
      training_record.push_back(_entry);
   }

   void clear()
   {
      the_stop_signal = 0;
      the_best_train_epoch = 0;
      the_best_train_perf = DBL_MAX;
      the_best_valid_epoch = 0;
      the_best_valid_perf = DBL_MAX;
      no_training_epochs = 0;
      training_record.clear();
   }

public:

   /* ***********************************
    *    Performance fields identifiers
    */

   class Entry
   {
   public:

      static const int train_grad_id = 1;
      static const int train_perf_id = 2;
      static const int valid_perf_id = 3;
      static const int test_perf_id = 4;

      Entry() {}

      Entry(long _epoch)
      {
         sample_epoch = _epoch;
      }

      long epoch() const
      {
         return sample_epoch;
      }

      void set_epoch(unsigned int _epoch)
      {
         sample_epoch = _epoch;
      }

      void set_training_gradient(double _val)
      {
         data_map[train_grad_id] = _val;
      }

      double training_gradient() const
      {
         return data_map.at(train_grad_id);
      }

      void set_training_perf(double _val)
      {
         data_map[train_perf_id] = _val;
      }

      double training_perf() const
      {
         return data_map.at(train_perf_id);
      }

      void set_validation_perf(double _val)
      {
         data_map[valid_perf_id] = _val;
      }

      double validation_perf() const
      {
         return data_map.at(valid_perf_id);
      }

      void set_test_perf(double _val)
      {
         data_map[test_perf_id] =_val;
      }

      double test_perf() const
      {
         return data_map.at(test_perf_id);
      }

      bool contains_key(unsigned int _id) const
      {
         return (data_map.find(_id) != data_map.end());
      }

      void clear()
      {
         sample_epoch = -1;
         data_map.clear();
      }

   private:
      long sample_epoch;
      map<unsigned int, double> data_map;
   };

private:

   vector<Entry> training_record;
   unsigned int the_stop_signal;
   unsigned int no_training_epochs;
   unsigned int the_best_train_epoch;
   double the_best_train_perf;
   unsigned int the_best_valid_epoch;
   double the_best_valid_perf;

};

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_TRAININGRECORD_H_ */
