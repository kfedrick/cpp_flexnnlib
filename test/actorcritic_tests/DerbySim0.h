//
// Created by kfedrick on 5/8/21.
//

#ifndef _DERBYSIM0_H_
#define _DERBYSIM0_H_

#include <flexnnet.h>
#include <Environment.h>
#include <RawFeatureSet.h>
#include <Reinforcement.h>
#include "SteeringActionFeature.h"

// Make room 10 by 1 so rep = 12 (terminal states on either side) x 2 (terminal state
// at end of room.
template<class State=RawFeature<14>, class Action=SteeringActionFeature, size_t N=1>
class DerbySim0 : public flexnnet::Environment<State, Action, N>
{
public:
   const unsigned int ROOM_WIDTH = 10;
   const unsigned int ROOM_LENGTH = 1;
   const unsigned int DOOR_POS = 10/2;
   const double GAIN{0.25};

public:
   DerbySim0();
   DerbySim0(const DerbySim0<State,Action,N>& _sim);
   virtual const flexnnet::Reinforcement<N>& get_reinforcement() const;
   virtual bool is_terminal(void) const;
   virtual const State& reset(void);

   virtual void set(unsigned int x, unsigned int y);

   const State& next(const typename Action::ActionEnum& _action);

   virtual const State& state() const;

   size_t size(void) const;

private:
   void update_state_vector() const;
   void encode_naive_state_vector() const;
   void encode_kcoded_state_vector() const;

private:
   unsigned int x_pos;
   unsigned int y_pos;

   mutable bool stale_state{true};
   mutable std::valarray<double> state_vector;
   mutable State variadic;

   mutable flexnnet::Reinforcement<N> reinforcement;

   mutable std::mt19937_64 rand_engine;
};

template<class S, class A, size_t N>
inline
DerbySim0<S,A,N>::DerbySim0()
{
   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);

   // Reset size of state vector
   state_vector.resize(ROOM_WIDTH+2+ROOM_LENGTH+1);

   std::get<0>(variadic.get_features()).decode(state_vector);
   reinforcement = flexnnet::Reinforcement<N>("R");
   //std::cout << " Constructor reinforcement : " << reinforcement.value_map().begin()->second.size() << "\n" << std::flush;
   //std::cout << "Constructor reinforcement name " << reinforcement.value_map().begin()->first << "\n";

   reset();

   //std::cout << "After Constructor reinforcement name " << reinforcement.value_map().begin()->first << "\n";

}

template<class S, class A, size_t N>
inline
DerbySim0<S,A,N>::DerbySim0(const DerbySim0<S,A,N>& _sim)
{
   x_pos = _sim.x_pos;
   y_pos = _sim.y_pos;
   stale_state = _sim.stale_state;
   state_vector = _sim.state_vector;
   variadic = _sim.variadic;
   reinforcement = _sim.reinforcement;
   rand_engine = _sim.rand_engine;

   //std::cout << "Copy Constructor reinforcement name " << reinforcement.value_map().begin()->first << "\n";

}

template<class S, class A, size_t N>
inline
size_t DerbySim0<S,A,N>::size(void) const
{
   return state_vector.size();
}

template<class S, class A, size_t N>
inline
const flexnnet::Reinforcement<N>& DerbySim0<S,A,N>::get_reinforcement() const
{
   double r;

   reinforcement.fill(-1.0);
   if (x_pos == 0 || x_pos == ROOM_WIDTH+1)
      reinforcement.set(0, -1.0);
   else if (y_pos >= ROOM_LENGTH)
   {
      if (x_pos > 4 && x_pos < 7)
         //if (x_pos == DOOR_POS || x_pos == DOOR_POS+1)
         reinforcement.set(0, 1.0);
      else
      {
         double dist = x_pos - (double) DOOR_POS;
         //reinforcement.set(0, exp(-GAIN*dist*dist));
         reinforcement.set(0, -1.0);
      }
   }
   //std::cout << " N : " << N << "\n" << std::flush;
   //std::cout << " reinforcement : " << reinforcement.value_map().begin()->second.size() << "\n" << std::flush;
   //std::cout << "get_reinforcement reinforcement name " << reinforcement.value_map().begin()->first << "\n";

   return reinforcement;
}

template<class S, class A, size_t N>
inline
bool DerbySim0<S,A,N>::is_terminal(void) const
{
   return (y_pos >= ROOM_LENGTH)
          || (x_pos == 0 || x_pos == ROOM_WIDTH+1);
}

template<class S, class A, size_t N>
inline
const S& DerbySim0<S,A,N>::reset(void)
{
   // Initialize position
   std::uniform_int_distribution<int> uniform_dist(1, ROOM_WIDTH);
   y_pos = 0;
   x_pos = uniform_dist(rand_engine);

   stale_state = true;
   return state();
}

template<class S, class A, size_t N>
inline
void DerbySim0<S,A,N>::set(unsigned int x, unsigned int y)
{
   x_pos = x;
   y_pos = y;
   stale_state = true;
}

template<class S, class A, size_t N>
inline
const S& DerbySim0<S,A,N>::next(const typename A::ActionEnum& _action)
{
   //std::cout << "DerbySim0::next " << x_pos << "\n" << std::flush;
   y_pos++;

   if (_action == A::ActionEnum::Left)
      x_pos--;
   else if (_action == A::ActionEnum::Right)
      x_pos++;

   //std::cout << "DerbySim0::next x_pos " << x_pos << "\n" << std::flush;


   stale_state = true;
   update_state_vector();

   return state();
}

template<class S, class A, size_t N>
inline
void DerbySim0<S,A,N>::update_state_vector() const
{
   //encode_naive_state_vector();
   encode_kcoded_state_vector();
}

template<class S, class A, size_t N>
inline
void DerbySim0<S,A,N>::encode_naive_state_vector() const
{
   if (!stale_state)
      return;

   state_vector = -1.0;
   state_vector[x_pos] = 1.0;
   state_vector[ROOM_WIDTH + 2 + y_pos] = 1.0;

   stale_state = false;
}

template<class S, class A, size_t N>
inline
void DerbySim0<S,A,N>::encode_kcoded_state_vector() const
{
   if (!stale_state)
      return;

   state_vector = -1.0;

   for (int i=0; i<=ROOM_WIDTH+1; i++)
      state_vector[i] = 2*exp(-abs(i-x_pos)) - 1;

   for (int i=0; i<=ROOM_LENGTH; i++)
      state_vector[i+ROOM_WIDTH + 2] = 2 * exp(-abs(i-y_pos)) - 1;

   stale_state = false;
}

template<class S, class A, size_t N>
inline
const S& DerbySim0<S,A,N>::state() const
{
   update_state_vector();

/*   for (int i=0; i<state_vector.size(); i++)
      std::cout << state_vector[i] << " ";
   std::cout << "\n";*/

   std::get<0>(variadic.get_features()).decode(state_vector);
   return variadic;
}

#endif //_DERBYSIM_H_
