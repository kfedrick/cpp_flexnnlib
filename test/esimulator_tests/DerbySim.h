//
// Created by kfedrick on 5/8/21.
//

#ifndef _DERBYSIM_H_
#define _DERBYSIM_H_

#include <flexnnet.h>
#include <Environment.h>
#include <RawFeatureSet.h>
#include <Reinforcement.h>
#include "SteeringAction.h"


class DerbySim : public flexnnet::Environment<flexnnet::RawFeatureSet<12>, SteeringAction, 1>
{
public:
   const unsigned int ROOM_WIDTH = 10;
   const unsigned int ROOM_LENGTH = 10;
   const unsigned int DOOR_POS = 10/2;
   const double GAIN{0.25};

public:
   DerbySim();
   virtual const flexnnet::Reinforcement<1>& get_reinforcement() const;
   virtual bool is_terminal(void) const;
   virtual const flexnnet::RawFeatureSet<12>& reset(void);

   const flexnnet::RawFeatureSet<12>& next(const SteeringActionFeature::ActionEnum& _action);

   virtual const flexnnet::RawFeatureSet<12>& state() const;

   size_t size(void) const;

private:
   void update_state_vector() const;

private:
   unsigned int x_pos;
   unsigned int y_pos;

   mutable bool stale_state{true};
   mutable std::valarray<double> state_vector;
   mutable flexnnet::RawFeatureSet<12> variadic;

   mutable flexnnet::Reinforcement<1> reinforcement;

   mutable std::mt19937_64 rand_engine;
};

DerbySim::DerbySim()
{
   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);

   // Reset size of state vector
   state_vector.resize(ROOM_WIDTH+2);

   std::get<0>(variadic.get_features()).decode(state_vector);
   reinforcement = flexnnet::Reinforcement<1>("position");

   reset();
}

size_t DerbySim::size(void) const
{
   return state_vector.size();
}

const flexnnet::Reinforcement<1>& DerbySim::get_reinforcement() const
{
   double r;

   reinforcement.fill(0);
   if (x_pos == 0 || x_pos == ROOM_WIDTH+1)
      reinforcement.set(0, -1.0);
   else // if (y_pos == ROOM_LENGTH-1)
   {
      if (x_pos == DOOR_POS)
         reinforcement.set(0, 1.0);
      else
      {
         double dist = x_pos - (double) DOOR_POS;
         reinforcement.set(0, exp(-GAIN*dist*dist));
      }
   }

   return reinforcement;
}

bool DerbySim::is_terminal(void) const
{
   return (y_pos >= ROOM_LENGTH - 1)
          || (x_pos == 0 || x_pos == ROOM_WIDTH+1);
}

const flexnnet::RawFeatureSet<12>& DerbySim::reset(void)
{
   // Initialize position
   std::uniform_int_distribution<int> uniform_dist(1, ROOM_WIDTH);
   y_pos = 0;
   x_pos = uniform_dist(rand_engine);

   stale_state = true;
   return state();
}

const flexnnet::RawFeatureSet<12>& DerbySim::next(const SteeringActionFeature::ActionEnum& _action)
{
   std::cout << "DerbySim::next " << x_pos << "\n" << std::flush;
   y_pos++;

   if (_action == SteeringActionFeature::ActionEnum::Left)
      x_pos--;
   else if (_action == SteeringActionFeature::ActionEnum::Right)
      x_pos++;

   std::cout << "DerbySim::next x_pos " << x_pos << "\n" << std::flush;


   stale_state = true;
   update_state_vector();

   return state();
}

void DerbySim::update_state_vector() const
{
   if (!stale_state)
      return;

   state_vector = -1.0;
   state_vector[x_pos] = 1.0;

   stale_state = false;
}

const flexnnet::RawFeatureSet<12>& DerbySim::state() const
{
   update_state_vector();
   std::get<0>(variadic.get_features()).decode(state_vector);
   return variadic;
}

#endif //_DERBYSIM_H_
