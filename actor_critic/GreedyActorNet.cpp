/*
 * GreedyActorNet.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: kfedrick
 */

#include <limits>
#include <NetSum.h>
#include <TanSig.h>
#include <GreedyActorNet.h>

using namespace std;

namespace flexnnet
{

   const Action GreedyActorNet::DEFAULT_ACTION;
   const vector<double> GreedyActorNet::DEFAULT_ACTION_PARAMS;

   GreedyActorNet::GreedyActorNet () : ActorNet ()
   {
      set<string> default_action_set;
      default_action_set.insert ("DO_NOTHING");

      set_stochastic_action_gain (1.0);
      init_actions (default_action_set);
   }

   GreedyActorNet::GreedyActorNet (const Pattern &stateVec, const set<string> &_actionSet, AdaptiveCriticNet *_adaptCritic)
      : ActorNet ()
   {
      //BaseLayer<NetSum, TanSig> &outputlayer = this->new_output_layer<NetSum, TanSig> (3,
      //                                                                             "output-layer");

      BaseLayer &outputlayer = this->new_output_layer<NetSum, TanSig> (3, "output-layer");


                                                                                   this->connect (stateVec, 0, outputlayer);

      outputlayer.init_weights ();

      critic_net = _adaptCritic;
      set_stochastic_action_gain (1.0);
      init_actions (_actionSet);
   }

   GreedyActorNet::~GreedyActorNet ()
   {
   }

   const Action &GreedyActorNet::get_action (const Pattern &_stateVec,
                                             unsigned int _recurStep)
   {
      unsigned int recurStep = 1;
      static Pattern actor_patt;
      double rsig;

      double best_rsig = -1000000.0;
      Action action;
      Action best_action = action_list.at (0);

      unsigned int sz = action_list.size ();
      vector<double> action_vec (sz);

      for (unsigned int i = 0; i < action_list.size (); i++)
      {
         action = action_list.at (i);

         action_vec.assign (sz, -1.0);
         action_vec.at (action.action_id () - 1) = 1.0;

         actor_patt = action_vec;

         rsig = critic_net->get_reinforcement (_stateVec, actor_patt, recurStep);
         if (rsig > best_rsig)
         {
            best_rsig = rsig;
            best_action = action;
         }
      }

      last_action = best_action;
      return last_action;
   }

   const Action &GreedyActorNet::get_stochastic_action (const Pattern &_stateVec, unsigned int _recurStep)
   {
      unsigned int recurStep = 1;
      static Pattern actor_patt;
      double rsig;

      double best_rsig = -100000.0;
      Action action;
      Action best_action = action_list.at (0);

      unsigned int sz = action_list.size ();
      vector<double> action_vec (sz);
      vector<double> exp_action_rsig_vec (sz);
      vector<double> action_prob_vec (sz);
      double sum_exp_rsig = 0;

      /*
       * Calculate probability vector
       */
      for (unsigned int i = 0; i < action_list.size (); i++)
      {
         action = action_list.at (i);

         action_vec.assign (sz, -1.0);
         action_vec.at (action.action_id () - 1) = 1.0;

         actor_patt = action_vec;

         rsig = critic_net->get_reinforcement (_stateVec, actor_patt, recurStep);

         exp_action_rsig_vec.at (i) = exp (stochastic_action_gain * rsig);
         sum_exp_rsig += exp_action_rsig_vec.at (i);

         if (rsig > best_rsig)
         {
            best_rsig = rsig;
            best_action = action;
         }
      }

      // softmax over reinforcement signals for each action
      for (unsigned int i = 0; i < sz; i++)
      {
         action_prob_vec.at (i) = exp_action_rsig_vec.at (i) / sum_exp_rsig;
      }

      // Calculate sort index of probability vector
      vector<unsigned int> sorted_indices (sz);
      for (unsigned int i = 0; i < sz; i++)
         sorted_indices[i] = i;

      unsigned int temp_ndx;
      for (unsigned int i = 0; i < sz - 1; i++)
      {
         for (unsigned int j = i + 1; j < sz; j++)
         {
            if (action_prob_vec.at (sorted_indices[j]) > action_prob_vec.at (sorted_indices[i]))
            {
               temp_ndx = sorted_indices[i];
               sorted_indices[i] = sorted_indices[j];
               sorted_indices[j] = temp_ndx;
            }
         }
      }

      // Find probabalistic action
      double threshold = 0;
      double rval = urand ();

      unsigned int ndx = 0;
      while (ndx < sz)
      {
         threshold += action_prob_vec.at (sorted_indices[ndx]);
         if (rval < threshold)
            break;

         ndx++;
      }

      best_action = action_list.at (sorted_indices[ndx]);

      last_action = best_action;
      return last_action;
   }

   const Action &GreedyActorNet::get_random_action ()
   {
      unsigned int sz = action_list.size ();

      last_action = action_list.at (urand (sz));
      return last_action;
   }

   void GreedyActorNet::init_actions (const set<string> &_actionSet)
   {
      action_list.clear ();

      unsigned int action_id = 0;

      set<string>::iterator actionset_iter;
      for (actionset_iter = _actionSet.begin (); actionset_iter != _actionSet.end (); actionset_iter++)
      {
         action_id++;

         const string &action_name = *actionset_iter;
         action_list.push_back (Action (action_name, action_id));
      }
   }

   const Pattern &GreedyActorNet::operator() ()
   {
      unsigned int sz = action_list.size ();
      vector<double> action_vec (sz);

      action_vec.assign (sz, -1.0);
      action_vec.at (last_action.action_id () - 1) = 1.0;

      best_actor_pattern = action_vec;

      return best_actor_pattern;
   }

   const Pattern &GreedyActorNet::raw ()
   {
      return raw ();
   }

/*
unsigned int GreedyActorNet::get_stochastic_action_index(const vector<double>& _actionProbVec)
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
*/

} /* namespace flexnnet */
