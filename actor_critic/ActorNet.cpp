/*
 * ActorNet.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#include <limits>
#include <NetSum.h>
#include <SoftMax.h>
#include <ActorNet.h>

using namespace std;

namespace flex_neuralnet
{

const Action ActorNet::DEFAULT_ACTION;
const vector<double> ActorNet::DEFAULT_ACTION_PARAMS;

ActorNet::ActorNet() : BaseNeuralNet("actor")
{
   set<string> default_action_set;
   default_action_set.insert("DO_NOTHING");

   set_stochastic_action_gain(1.0);
   init_actions(default_action_set);
}

ActorNet::ActorNet(const set<string>& _actionSet) : BaseNeuralNet("actor")
{
   /*
   Layer<NetSum, SoftMax>& outputlayer =
         this->new_output_layer<NetSum, SoftMax>(_actionSet.size(),
               "actor-output");
               */

   set_stochastic_action_gain(1.0);
   init_actions(_actionSet);
}

ActorNet::~ActorNet()
{
}

const Action& ActorNet::get_action(const Pattern& _stateVec,
      unsigned int _recurStep)
{
   BaseNeuralNet* net = dynamic_cast<BaseNeuralNet*>(this);
   static Pattern actor_opatt;

   actor_opatt = (*net)(_stateVec, _recurStep);

   if (actor_opatt.size() == 0)
      return ActorNet::DEFAULT_ACTION;

   unsigned int max_unit_ndx = 0;
   double max_unit_val = actor_opatt().at(max_unit_ndx);

   for (unsigned int patt_ndx = 1; patt_ndx < actor_opatt().size(); patt_ndx++)
   {
      if (actor_opatt()[patt_ndx] > max_unit_val)
      {
         max_unit_ndx = patt_ndx;
         max_unit_val = actor_opatt()[max_unit_ndx];
      }
   }

   last_action = action_list.at(max_unit_ndx);

   return last_action;
}

const Action& ActorNet::get_stochastic_action(const Pattern& _stateVec, unsigned int _recurStep)
{
   BaseNeuralNet* net = dynamic_cast<BaseNeuralNet*>(this);
   static Pattern actor_opatt;
   Action best_action = action_list.at(0);

   unsigned int sz = action_list.size();

   vector<double> action_vec(sz);
   vector<double> exp_action_opatt(sz);
   vector<double> action_prob_vec(sz);
   double sum_exp_action_opatt = 0;

   actor_opatt = (*net)(_stateVec, _recurStep);

   if (actor_opatt.size() == 0)
      return ActorNet::DEFAULT_ACTION;

   /*
    * Calculate probability vector
    */
   for (unsigned int i=0; i<sz; i++)
   {
      exp_action_opatt.at(i) = exp(1.0 * stochastic_action_gain * actor_opatt().at(i));
      sum_exp_action_opatt += exp_action_opatt.at(i);
   }

   // softmax over each action potential
   for (unsigned int i=0; i<sz; i++)
   {
      action_prob_vec.at(i) = exp_action_opatt.at(i) / sum_exp_action_opatt;
   }

   // Calculate sort index of probability vector
   vector<unsigned int> sorted_indices(sz);
   for (unsigned int i=0; i<sz; i++)
       sorted_indices[i] = i;

   unsigned int temp_ndx;
   for (unsigned int i=0; i<sz-1; i++)
   {
      for (unsigned int j=i+1; j<sz; j++)
      {
         if (action_prob_vec.at(sorted_indices[j]) > action_prob_vec.at(sorted_indices[i]))
         {
            temp_ndx = sorted_indices[i];
            sorted_indices[i] = sorted_indices[j];
            sorted_indices[j] = temp_ndx;
         }
      }
   }

   // Find probabalistic action
   double threshold = 0;
   double rval = urand();

   unsigned int ndx = 0;
   while (ndx < sz-1)
   {
      threshold += action_prob_vec.at(sorted_indices[ndx]);
      if (rval < threshold)
         break;

      ndx++;
   }

   best_action = action_list.at(sorted_indices[ndx]);

   last_action = best_action;
   return last_action;
}

void ActorNet::init_actions(const set<string>& _actionSet)
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

unsigned int ActorNet::get_stochastic_action_index(const vector<double>& _actionProbVec)
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

const Pattern& ActorNet::operator()()
{
   unsigned int sz = action_list.size();
   vector<double> action_vec(sz);

   action_vec.assign(sz, -1.0);
   action_vec.at(last_action.action_id()-1) = 1.0;

   best_actor_pattern = action_vec;

   return best_actor_pattern;
}


const Pattern& ActorNet::raw()
{
   BaseNeuralNet* ptr = dynamic_cast<BaseNeuralNet*>(this);
   return (*ptr)();
}

} /* namespace flex_neuralnet */
