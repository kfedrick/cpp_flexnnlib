/*
 * GreedyActorNet.h
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_GREEDYACTOR_NET_H_
#define FLEX_NEURALNET_GREEDYACTOR_NET_H_

#include <vector>
#include <set>
#include <cstdlib>

#include "ActorNet.h"
#include "AdaptiveCriticNet.h"
#include "BaseNeuralNet.h"
#include "Action.h"

using namespace std;

namespace flexnnet
{

   class GreedyActorNet : public ActorNet
   {
   public:
      static const Action DEFAULT_ACTION;
      static const int DEFAULT_ACTION_ID = Action::NO_ACTION;
      static const vector<double> DEFAULT_ACTION_PARAMS;

   public:
      GreedyActorNet ();
      GreedyActorNet (const Pattern &stateVec, const set<string> &_actionSet, AdaptiveCriticNet *_adaptCritic);
      virtual ~GreedyActorNet ();

      const Pattern &operator() ();
      const Pattern &raw ();

      virtual const Action &get_action (const Pattern &_stateVec, unsigned int _recurStep = 1);
      virtual const Action &get_stochastic_action (const Pattern &_stateVec, unsigned int _recurStep = 1);
      virtual const Action &get_random_action ();
      const vector<Action> &get_action_list ();

   protected:
      void init_actions (const set<string> &_actionSet);

      unsigned int get_stochastic_action_index (const vector<double> &_actionProbVec);
      double urand ();
      int urand (int u);

   private:
      AdaptiveCriticNet *critic_net;
      vector<Action> action_list;
      Action last_action;
      Pattern best_actor_pattern;
   };

// Generate a random number between 0 and 1
// return a uniform number in [0,1].
   inline
   double GreedyActorNet::urand ()
   {
      return rand () / double (RAND_MAX);
   }

   inline
   int GreedyActorNet::urand (int u)
   {
      unsigned int top = ((((RAND_MAX - u) + 1) / u) * u - 1) + u;
      unsigned int r;
      do
      {
         r = rand ();
      }
      while (r > top);
      return (r % u);
   }

   inline
   const vector<Action> &GreedyActorNet::get_action_list ()
   {
      return action_list;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_GREEDYACTOR_NET_H_ */
