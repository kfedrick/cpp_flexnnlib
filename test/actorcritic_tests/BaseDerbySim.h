//
// Created by kfedrick on 5/8/21.
//

#ifndef _DERBYSIM2_H_
#define _DERBYSIM2_H_

#include <flexnnet.h>
#include <Environment.h>
#include <RawFeatureSet.h>
#include "CMapState.h"
#include <Reinforcement.h>
#include "SteeringActionFeature.h"

// Make room 10 by 10 so rep = 12 (terminal states on either side) x 11 (terminal state
// at end of room.
template<class State=CMapState, class Action=SteeringActionFeature, size_t N=1>
class BaseDerbySim : public flexnnet::Environment<flexnnet::RawFeature<W+2+L+1>, Action, N>
{
public:
   const unsigned int ROOM_WIDTH = 10;
   const unsigned int ROOM_LENGTH = 10;
   const unsigned int DOOR_POS = 10/2;
   const double GAIN{0.25};

   //using State = RawFeatureSet<W+2+L+1>;

public:
   BaseDerbySim();
   BaseDerbySim(const BaseDerbySim<W,L,Action,N>& _sim);
   virtual const flexnnet::Reinforcement<N>& get_reinforcement() const;
   virtual bool is_terminal(void) const;
   virtual const State& reset(void);

   const State& next(const typename Action::ActionEnum& _action);

   virtual const State& state() const;

   size_t size(void) const;

private:
   void update_state_vector() const;
   void encode_naive_state_vector() const;
   void encode_kcoded_state_vector() const;
   void encode_cmap_state_vector() const;

private:
   unsigned int x_pos;
   unsigned int y_pos;

   mutable bool stale_state{true};
   mutable std::valarray<double> state_vector;
   mutable State variadic;

   mutable flexnnet::Reinforcement<N> reinforcement;

   mutable std::mt19937_64 rand_engine;
};

template<size_t W, size_t L, class A, size_t N>
inline
BaseDerbySim<W,L,A,N>::BaseDerbySim()
{
   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);

   // Reset size of state vector
   state_vector.resize((W+2)*(L+1));

   std::get<0>(variadic.get_features()).decode(state_vector);
   reinforcement = flexnnet::Reinforcement<N>("R");
   //std::cout << " Constructor reinforcement : " << reinforcement.value_map().begin()->second.size() << "\n" << std::flush;
   //std::cout << "Constructor reinforcement name " << reinforcement.value_map().begin()->first << "\n";

   reset();

   //std::cout << "After Constructor reinforcement name " << reinforcement.value_map().begin()->first << "\n";

}

template<size_t W, size_t L, class A, size_t N>
inline
BaseDerbySim<W,L,A,N>::BaseDerbySim(const BaseDerbySim<S,A,N>& _sim)
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

template<size_t W, size_t L, class A, size_t N>
inline
size_t BaseDerbySim<W,L,A,N>::size(void) const
{
   return state_vector.size();
}

template<size_t W, size_t L, class A, size_t N>
inline
const flexnnet::Reinforcement<N>& BaseDerbySim<W,L,A,N>::get_reinforcement() const
{
   double r;

   reinforcement.fill(0);
   if (x_pos == 0 || x_pos == ROOM_WIDTH+1)
      reinforcement.set(0, -1.0);
   else if (y_pos >= ROOM_LENGTH)
   {
      if (x_pos > 3 && x_pos < 9)
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

template<size_t W, size_t L, class A, size_t N>
inline
bool BaseDerbySim<W,L,A,N>::is_terminal(void) const
{
   return (y_pos > L)
          || (x_pos == 0 || x_pos == W+1);
}

template<size_t W, size_t L, class A, size_t N>
inline
const typename BaseDerbySim<W,L,A,N>::State& BaseDerbySim<W,L,A,N>::reset(void)
{
   // Initialize position
   std::uniform_int_distribution<int> uniform_dist(1, ROOM_WIDTH);
   y_pos = 0;
   x_pos = uniform_dist(rand_engine);

   stale_state = true;
   return state();
}

template<size_t W, size_t L, class A, size_t N>
inline
const typename BaseDerbySim<W,L,A,N>::State& BaseDerbySim<W,L,A,N>::next(const typename A::ActionEnum& _action)
{
   //std::cout << "BaseDerbySim::next " << x_pos << "\n" << std::flush;
   y_pos++;

   if (_action == A::ActionEnum::Left)
      x_pos--;
   else if (_action == A::ActionEnum::Right)
      x_pos++;

   //std::cout << "BaseDerbySim::next x_pos " << x_pos << "\n" << std::flush;


   stale_state = true;
   update_state_vector();

   return state();
}

template<size_t W, size_t L, class A, size_t N>
inline
void BaseDerbySim<W,L,A,N>::update_state_vector() const
{
   //encode_naive_state_vector();
   encode_kcoded_state_vector();
}

template<size_t W, size_t L, class A, size_t N>
inline
void BaseDerbySim<W,L,A,N>::encode_naive_state_vector() const
{
   if (!stale_state)
      return;

   state_vector = -1.0;
   state_vector[x_pos] = 1.0;
   state_vector[W + 2 + y_pos] = 1.0;

   stale_state = false;
}

template<size_t W, size_t L, class A, size_t N>
inline
void BaseDerbySim<W,L,A,N>::encode_kcoded_state_vector() const
{
   if (!stale_state)
      return;

   state_vector = -1.0;

   for (int i=0; i<=W; i++)
      state_vector[i] = 2*exp(-abs(i-x_pos)) - 1;

   for (int i=0; i<=L; i++)
      state_vector[i+W + 1] = 2 * exp(-abs(i-y_pos)) - 1;

   stale_state = false;
}

template<size_t W, size_t L, class A, size_t N>
inline
void BaseDerbySim<W,L,A,N>::encode_cmap_state_vector() const
{
   if (!stale_state)
      return;

   state_vector = -1.0;

   for (int y=0; y <= L; y++)
      for (int x=0; x <= W + 1; x++)
         state_vector[y * W + 2 + x] = 2 * exp(-abs(x - x_pos + y - y_pos)) - 1;

   stale_state = false;
}

template<size_t W, size_t L, class A, size_t N>
inline
const typename BaseDerbySim<W,L,A,N>::State& BaseDerbySim<W,L,A,N>::state() const
{
   update_state_vector();

/*   for (int i=0; i<state_vector.size(); i++)
      std::cout << state_vector[i] << " ";
   std::cout << "\n";*/

   std::get<0>(variadic.get_features()).decode(state_vector);
   return variadic;
}

#endif //_DERBYSIM2_H_
