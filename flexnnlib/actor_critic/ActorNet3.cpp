/*
 * ActorNet3.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#include <limits>
#include <NetSum.h>
#include <SoftMax.h>
#include <ActorNet3.h>

using namespace std;

namespace flex_neuralnet
{

const Action ActorNet3::DEFAULT_ACTION;
const vector<double> ActorNet3::DEFAULT_ACTION_PARAMS;

ActorNet3::ActorNet3() : TDCNeuralNet("actor")
{
   set<string> default_action_set;
   default_action_set.insert("DO_NOTHING");

   set_stochastic_action_var(1.0);
   init_actions(default_action_set);
}

ActorNet3::ActorNet3(const set<string>& _actionSet) : TDCNeuralNet("actor")
{
   /*
   Layer<NetSum, SoftMax>& outputlayer =
         this->new_output_layer<NetSum, SoftMax>(_actionSet.size(),
               "actor-output");
               */

   set_stochastic_action_var(1.0);
   init_actions(_actionSet);
}

ActorNet3::~ActorNet3()
{
}

const Action& ActorNet3::get_action(const Pattern& _stateVec,
      unsigned int _recurStep)
{
   TDCNeuralNet* net = dynamic_cast<TDCNeuralNet*>(this);
   static Pattern actor_opatt;

   actor_opatt = (*net)(_stateVec, _recurStep);

   if (actor_opatt.size() == 0)
      return ActorNet3::DEFAULT_ACTION;

   double aval = actor_opatt().at(0);

   //cout << "actor out " << aval << endl;

   vector<double> rvec = {aval};
   urand_raw_pattern = rvec;


   unsigned int sz = action_list.size();
   double delta = 2.0 / sz;
   double bound = -1.0 + delta;
   unsigned int ndx;

   for (ndx = 0; ndx<sz-1 && aval > bound; ndx++, bound+=delta);

   last_action = action_list.at(ndx);

   return last_action;

   /*
   get_stochastic_action(_stateVec, _recurStep);
   */
}

const Action& ActorNet3::get_stochastic_action(const Pattern& _stateVec, unsigned int _recurStep)
{
   TDCNeuralNet* net = dynamic_cast<TDCNeuralNet*>(this);
   static Pattern actor_opatt;

   actor_opatt = (*net)(_stateVec, _recurStep);

   if (actor_opatt.size() == 0)
      return ActorNet3::DEFAULT_ACTION;

   double aval = actor_opatt().at(0);

   //cout << "actor out " << aval << endl;

   unsigned int sz = action_list.size();
   double delta = 2.0 / sz;
   double bound = -1.0 + delta;
   unsigned int ndx;

   // get gaussian random variable with mean aval and 2 stddev
   // as the bin width
   double rval = nrand(aval, stochastic_action_var * delta);

   vector<double> rvec = {rval};
   urand_raw_pattern = rvec;

   /*
   // Uniform random will result in completely random moves
   double rval = 2.0 * urand() - 1.0;
   vector<double> rvec = {rval};
   urand_raw_pattern = rvec;
   */

   for (ndx = 0; ndx<sz-1 && rval > bound; ndx++, bound+=delta);

   last_action = action_list.at(ndx);

   return last_action;
}



void ActorNet3::init_actions(const set<string>& _actionSet)
{
   action_list.clear();

   unsigned int action_id = 0;

   set<string>::iterator actionset_iter;
   for (actionset_iter = _actionSet.begin(); actionset_iter != _actionSet.end(); actionset_iter++)
   {
      action_id++;

      const string& action_name = *actionset_iter;
      action_list.push_back( Action(action_name, action_id) );
   }
}

const Action& ActorNet3::get_action_by_index(unsigned int _aNdx)
{
   return action_list.at(_aNdx);
}

unsigned int ActorNet3::get_stochastic_action_index(const vector<double>& _actionProbVec)
{
   double threshold = 0;
   double rval = urand();

   unsigned int ndx = 0;
   for (unsigned int i=0; i<_actionProbVec.size(); i++)
   {
      ndx++;

      threshold += _actionProbVec.at(i);
      if (rval < threshold)
         break;
   }

   return ndx-1;
}

unsigned int ActorNet3::get_action_index(const vector<double>& _actionProbVec)
{
   unsigned int sz = action_list.size();
   double delta = 2.0 / sz;
   double bound = -1.0 + delta;
   unsigned int ndx;

   double aval = _actionProbVec.at(0);
   for (ndx = 0; ndx<sz-1 && aval > bound; ndx++, bound+=delta);

   return ndx;
}


const Pattern& ActorNet3::operator()()
{
   //return raw();
   return urand_raw_pattern;
}


const Pattern& ActorNet3::raw()
{
   //BaseNeuralNet* ptr = dynamic_cast<BaseNeuralNet*>(this);
   //return (*ptr)();

   return urand_raw_pattern;
}

} /* namespace flex_neuralnet */
