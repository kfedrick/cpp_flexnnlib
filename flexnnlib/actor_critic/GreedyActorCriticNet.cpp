/*
 * GreedyActorCriticNet.cpp
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#include "GreedyActorCriticNet.h"

namespace flex_neuralnet
{

GreedyActorCriticNet::GreedyActorCriticNet(const Pattern& stateVec, const set<string>& _actionSet, AdaptiveCriticNet* _adaptCritic)
{
   adaptive_critic = _adaptCritic;
   actor = new GreedyActorNet(stateVec, _actionSet, _adaptCritic);
   set_stochastic_action(true);
}

GreedyActorCriticNet::~GreedyActorCriticNet()
{
   // TODO Auto-generated destructor stub
}

ActorNet* GreedyActorCriticNet::get_actor()
{
   return actor;
}

AdaptiveCriticNet* GreedyActorCriticNet::get_adaptive_critic()
{
   return adaptive_critic;
}

/*
 * Activate actor-critic networks for the specified Pattern representing
 * the environment state vector.
 */
const ActorCriticOutput& GreedyActorCriticNet::operator()(const Pattern& ipattern, unsigned int recurStep)
{
   Pattern actor_opatt;
   Action actor_action;

   Pattern adaptive_critic_opatt;
   double rsig;

   /*
    * Activate actor and get corresponding action using the
    * ActorNetwork operator(). Then get the raw actor output
    * pattern from the last activation using the native NeuralNet
    * operator() method so we can pass it to the actor-critic.
    */
   if (stochastic_action)
      actor_action = actor->get_random_action();
      //actor_action = actor->get_stochastic_action(ipattern, recurStep);
   else
      actor_action = actor->get_action(ipattern, recurStep);

   actor_opatt = (*actor)();

   // Activate adaptive critic and get output (reinforcement, rflag)
   rsig = adaptive_critic->get_reinforcement(ipattern, actor_opatt, recurStep);

   // Set local cached value for actor-critic model output
   last_actor_critic_output.set(actor_action, rsig, true);

   return last_actor_critic_output;
}


void GreedyActorCriticNet::clear_error(unsigned int timeStep)
{
   actor->clear_error(timeStep);
   adaptive_critic->clear_error(timeStep);
}

void GreedyActorCriticNet::backprop(const vector<double>& _eVec, unsigned int timeStep)
{
   adaptive_critic->backprop(_eVec, timeStep);
   //const vector<BaseLayer*> layers = actor->get_network_layers();

   const vector< vector<double> >& critic_bp_errvec = adaptive_critic->get_input_error();

   /*
   cout << "----- critic bp errorv -----" << endl;
   for (unsigned int vec_ndx=0; vec_ndx<actor_bp_errvec.size(); vec_ndx++)
   {
      const vector<double>& avec = actor_bp_errvec.at(vec_ndx);
      for (unsigned int ndx=0; ndx<avec.size(); ndx++)
      {
         cout << avec.at(ndx) << " ";
      }
      cout << endl;
   }
   cout << "----------------" << endl;
   */

   /*
    * For kicks, negate the error vector. I may have a sign issue. I've
    * never done a backprop between networks before.
    */
   vector<double> net_bp_errorv(critic_bp_errvec.at(1).size());
   for (unsigned int i=0; i<critic_bp_errvec.at(1).size(); i++)
      net_bp_errorv.at(i) = critic_bp_errvec.at(1).at(i);

   actor->backprop(net_bp_errorv, timeStep);
}

} /* namespace flex_neuralnet */
