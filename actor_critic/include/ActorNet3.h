/*
 * ActorNet3.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_ACTOR_NET3_H_
#define FLEX_NEURALNET_ACTOR_NET3_H_

#include <vector>
#include <set>
#include <cstdlib>
#include <cmath>

#include "TDCNeuralNet.h"
#include "Action.h"

using namespace std;

namespace flex_neuralnet
{

class ActorNet3 : public TDCNeuralNet
{
public:
   static const Action DEFAULT_ACTION;
   static const int DEFAULT_ACTION_ID = Action::NO_ACTION;
   static const vector<double> DEFAULT_ACTION_PARAMS;

public:
   ActorNet3();
   ActorNet3(const set<string>& _actionSet);
   virtual ~ActorNet3();

   double get_stochastic_action_var();
   void set_stochastic_action_var(const double _val);

   virtual const Action& get_action(const Pattern& _stateVec, unsigned int _recurStep = 1);
   virtual const Action& get_stochastic_action(const Pattern& _stateVec, unsigned int _recurStep = 1);
   const vector<Action>& get_action_list();
   const Pattern& operator()();
   const Pattern& raw();

   const Action& get_action_by_index(unsigned int _aNdx);
   unsigned int get_action_index(const vector<double>& _actionProbVec);
   unsigned int get_stochastic_action_index(const vector<double>& _actionProbVec);

protected:
   void init_actions(const set<string>& _actionSet);

   double urand();
   double nrand(double mean, double stdev);

   double stochastic_action_var;

private:
   vector<Action> action_list;
   Action last_action;
   Pattern best_actor_pattern;
   Pattern urand_raw_pattern;
};

inline
double ActorNet3::get_stochastic_action_var()
{
   return stochastic_action_var;
}

inline
void ActorNet3::set_stochastic_action_var(const double _val)
{
   stochastic_action_var = _val;
}

// Generate a random number between 0 and 1
// return a uniform number in [0,1].
inline
double ActorNet3::urand()
{
   return rand() / double(RAND_MAX);
}

inline
double ActorNet3::nrand(double mean, double stdev)
{
   double pi = 3.1415926535897;
   double r1, r2;
   r1 = urand();
   r2 = urand();
   return mean + stdev * sqrt(-2 * log(r1)) * cos(2 * pi * r2);
}

inline
const vector<Action>& ActorNet3::get_action_list()
{
   return action_list;
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_ACTOR_NET3_H_ */
