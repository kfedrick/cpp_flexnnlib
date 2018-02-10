/*
 * PBActorNet.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#include <limits>
#include <NetSum.h>
#include <SoftMax.h>
#include <PBActorNet.h>

using namespace std;

namespace flex_neuralnet
{

const Action PBActorNet::DEFAULT_ACTION;
const vector<double> PBActorNet::DEFAULT_ACTION_PARAMS;

PBActorNet::PBActorNet() : TDCNeuralNet("actor")
{
   set<string> default_action_set;
   default_action_set.insert("NONE");

   set_stochastic_action_var(1.0);
   init_actions(default_action_set);
}

PBActorNet::PBActorNet(const set<string>& _actionSet) : TDCNeuralNet("actor")
{

   set_stochastic_action_var(1.0);
   init_actions(_actionSet);
}

PBActorNet::~PBActorNet()
{
}

const Action& PBActorNet::get_action(const Pattern& _stateVec,
      unsigned int _recurStep)
{
   TDCNeuralNet* net = dynamic_cast<TDCNeuralNet*>(this);
   static Pattern actor_opatt;

   actor_opatt = (*net)(_stateVec, _recurStep);

   if (actor_opatt.size() == 0)
      return PBActorNet::DEFAULT_ACTION;

   double aval = actor_opatt().at(0);

   //cout << "actor out " << aval << endl;

   vector<double> rvec = {aval};
   urand_raw_pattern = rvec;

   if (aval > 0)
      last_action = action_list.at(0);
   else
      last_action = action_list.at(1);

   return last_action;
}

const Action& PBActorNet::get_stochastic_action(const Pattern& _stateVec, unsigned int _recurStep)
{
   TDCNeuralNet* net = dynamic_cast<TDCNeuralNet*>(this);
   static Pattern actor_opatt;

   actor_opatt = (*net)(_stateVec, _recurStep);

   if (actor_opatt.size() == 0)
      return PBActorNet::DEFAULT_ACTION;

   double aval = actor_opatt().at(0);

   //cout << "actor out " << aval << endl;

   vector<double> rvec = {aval};
   urand_raw_pattern = rvec;

   if (aval > 0)
      last_action = action_list.at(0);
   else
      last_action = action_list.at(1);

   return last_action;
}



void PBActorNet::init_actions(const set<string>& _actionSet)
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

const Action& PBActorNet::get_action_by_index(unsigned int _aNdx)
{
   return action_list.at(_aNdx);
}

unsigned int PBActorNet::get_stochastic_action_index(const vector<double>& _actionProbVec)
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

unsigned int PBActorNet::get_action_index(const vector<double>& _actionProbVec)
{
   unsigned int sz = action_list.size();
   double delta = 2.0 / sz;
   double bound = -1.0 + delta;
   unsigned int ndx;

   double aval = _actionProbVec.at(0);
   for (ndx = 0; ndx<sz-1 && aval > bound; ndx++, bound+=delta);

   return ndx;
}


const Pattern& PBActorNet::operator()()
{
   //return raw();
   return urand_raw_pattern;
}


const Pattern& PBActorNet::raw()
{
   //BaseNeuralNet* ptr = dynamic_cast<BaseNeuralNet*>(this);
   //return (*ptr)();

   return urand_raw_pattern;
}

} /* namespace flex_neuralnet */
