//
// Created by kfedrick on 5/8/21.
//

#ifndef _DERBYSIM_H_
#define _DERBYSIM_H_

#include <flexnnet.h>
#include <ESimulator.h>
#include <Environment.h>
#include <VariadicNetworkInput.h>
#include <EnvironReinforcement.h>
#include "Reinforcement.h"

enum class ActionEnum { Left, Right };

class DerbySim : public flexnnet::Environment<flexnnet::VariadicNetworkInput<std::valarray<double>>, ActionEnum, 1>
{
public:
   const unsigned int ROOM_WIDTH = 10;
   const unsigned int ROOM_LENGTH = 10;
   const unsigned int DOOR_POS = 10/2;
   const double GAIN{0.25};

public:
   DerbySim();
   virtual const flexnnet::EnvironReinforcement<1>& get_reinforcement() const;
   virtual bool is_terminal(void) const;
   virtual const flexnnet::VariadicNetworkInput<std::valarray<double>>& reset(void);

   const flexnnet::VariadicNetworkInput<std::valarray<double>>& next(const ActionEnum& _action);

   virtual const flexnnet::VariadicNetworkInput<std::valarray<double>>& state() const;

   size_t size(void) const;

private:
   void update_state_vector() const;

private:
   unsigned int x_pos;
   unsigned int y_pos;

   mutable bool stale_state{true};
   mutable std::tuple<std::valarray<double>> state_vector;
   mutable flexnnet::VariadicNetworkInput<std::valarray<double>> variadic;

   mutable flexnnet::EnvironReinforcement<1> reinforcement;

   mutable std::mt19937_64 rand_engine;
};

DerbySim::DerbySim()
{
   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);

   // Reset size of state vector
   std::get<0>(state_vector) = std::valarray<double>(ROOM_WIDTH + 2);

   auto fp = std::pair<std::string, std::valarray<double>>("position", std::get<0>(state_vector));
   variadic = flexnnet::VariadicNetworkInput<std::valarray<double>>(fp);
   reinforcement = flexnnet::EnvironReinforcement<1>({"position"});

   reset();
}

size_t DerbySim::size(void) const
{
   return std::get<0>(state_vector).size();
}

const flexnnet::EnvironReinforcement<1>& DerbySim::get_reinforcement() const
{
   double r;

   reinforcement.fill(0);
   if (x_pos == 0 || x_pos == ROOM_WIDTH+1)
      reinforcement[0] = -1.0;
   else // if (y_pos == ROOM_LENGTH-1)
   {
      if (x_pos == DOOR_POS)
         reinforcement[0] = 1.0;
      else
      {
         double dist = x_pos - (double) DOOR_POS;
         reinforcement[0] = exp(-GAIN*dist*dist);
      }
   }

   return reinforcement;
}

bool DerbySim::is_terminal(void) const
{
   return (y_pos >= ROOM_LENGTH - 1)
          || (x_pos == 0 || x_pos == ROOM_WIDTH+1);
}

const flexnnet::VariadicNetworkInput<std::valarray<double>>& DerbySim::reset(void)
{
   // Initialize position
   std::uniform_int_distribution<int> uniform_dist(1, ROOM_WIDTH);
   y_pos = 0;
   x_pos = uniform_dist(rand_engine);

   stale_state = true;
   return state();
}

const flexnnet::VariadicNetworkInput<std::valarray<double>>& DerbySim::next(const ActionEnum& _action)
{
   std::cout << "DerbySim::next " << x_pos << "\n" << std::flush;
   y_pos++;

   if (_action == ActionEnum::Left)
      x_pos--;
   else if (_action == ActionEnum::Right)
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

   std::get<0>(state_vector) = -1.0;
   std::get<0>(state_vector)[x_pos] = 1.0;

   stale_state = false;
}

const flexnnet::VariadicNetworkInput<std::valarray<double>>& DerbySim::state() const
{
   update_state_vector();
   variadic.set(state_vector);
   return variadic;
}

#endif //_DERBYSIM_H_
