/*
 * ActorCriticNet2.cpp
 *
 *  Created on: Mar 21, 2015
 *      Author: kfedrick
 */

#include "ActorCriticNet2.h"

namespace flex_neuralnet
{

ActorCriticNet2::ActorCriticNet2(ActorNet2* _actor,
      AdaptiveCriticNet* _adaptCritic)
{
   actor = _actor;
   adaptive_critic = _adaptCritic;
   set_stochastic_action(true);
}

ActorCriticNet2::~ActorCriticNet2()
{
   // TODO Auto-generated destructor stub
}

ActorNet2* ActorCriticNet2::get_actor()
{
   return actor;
}

AdaptiveCriticNet* ActorCriticNet2::get_adaptive_critic()
{
   return adaptive_critic;
}

/*
 * Activate actor-critic networks for the specified Pattern representing
 * the environment state vector.
 */
const ActorCriticOutput& ActorCriticNet2::operator()(const Pattern& ipattern,
      unsigned int recurStep)
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
      actor_action = actor->get_stochastic_action(ipattern, recurStep);
   else
      actor_action = actor->get_action(ipattern, recurStep);

   actor_opatt = (*actor)();

   /*
   if (print_gradient)
   {
      printf( "actor opatt = %f\n", actor_opatt().at(0));
   }
   */

   // Activate adaptive critic and get output (reinforcement, rflag)
   rsig = adaptive_critic->get_reinforcement(ipattern, actor_opatt, recurStep);

   // Set local cached value for actor-critic model output
   last_actor_critic_output.set(actor_action, rsig, true);

   return last_actor_critic_output;
}

void ActorCriticNet2::clear_error(unsigned int timeStep)
{
   actor->clear_error(timeStep);
   adaptive_critic->clear_error(timeStep);
}

void ActorCriticNet2::backprop(const vector<double>& _eVec,
      unsigned int timeStep)
{
   adaptive_critic->backprop(_eVec, timeStep);
   //const vector<BaseLayer*> layers = actor->get_network_layers();

   const vector<BaseLayer*> network_layers = adaptive_critic->get_network_layers();
   BaseLayer& layer = *network_layers[0];
   string name = layer.name();
   const Array<double>& dAdN = layer.get_dAdN();
   const Array<double>& dEdW = layer.get_dEdW();

   if (print_gradient)
   {
      cout << "----- critic dAdN (" << layer.name() << ") -----" << endl;
      for (unsigned int ondx = 0; ondx < dAdN.rowDim(); ondx++)
      {
         for (unsigned int indx = 0; indx < dAdN.colDim(); indx++)
            printf("%6.3f ", dAdN[ondx][indx]);
         printf("\n");
      }
      cout << endl;
      cout << "----------------" << endl;

      cout << "----- critic dEdW (" << layer.name() << ") -----" << endl;
      for (unsigned int ondx = 0; ondx < dEdW.rowDim(); ondx++)
      {
         for (unsigned int indx = 0; indx < dEdW.colDim(); indx++)
            printf("%6.3f ", dEdW[ondx][indx]);
         printf("\n");
      }
      cout << endl;
      cout << "----------------" << endl;
   }


   const vector<vector<double> >& critic_bp_errvec =
         adaptive_critic->get_input_error();

   if (print_gradient)
   {
      cout << "----- critic bp input gradientv -----" << endl;
      const vector<double>& avec = critic_bp_errvec.at(1);
      for (unsigned int ndx = 0; ndx < avec.size(); ndx++)
      {
         cout << avec.at(ndx) << " ";
      }
      cout << endl;
      cout << "----------------" << endl;
   }

   /*
    * Error vector 1 for the critic is for the actor input. vector 0
    * is for the state
    */
   vector<double> net_bp_errorv(critic_bp_errvec.at(1).size());
   for (unsigned int i = 0; i < critic_bp_errvec.at(1).size(); i++)
      net_bp_errorv.at(i) = critic_bp_errvec.at(1).at(i);

   actor->backprop(net_bp_errorv, timeStep);

   /*
   cout << "----- actor out dAdN -----" << endl;
   int oindx = actor->get_network_layers().size();
   const Array<double>& dAdN = actor->layer(oindx-1).get_dAdN();
   for (unsigned int i=0; i<dAdN.rowDim(); i++)
   {
      for (unsigned int j=0; j<dAdN.colDim(); j++)
      {
         cout << dAdN.at(i,j) << " ";
      }
      cout << endl;
   }
   cout << "-----------------------" << endl;

   cout << "----- actor out dEdW -----" << endl;
   const Array<double>& dEdW = actor->layer(oindx-1).get_dEdW();
   for (unsigned int i=0; i<dEdW.rowDim(); i++)
   {
      for (unsigned int j=0; j<dEdW.colDim(); j++)
      {
         cout << dEdW.at(i,j) << " ";
      }
      cout << endl;
   }
   cout << "-----------------------" << endl;

   cout << "----- actor out dNdW -----" << endl;
   const Array<double>& dNdW = actor->layer(oindx-1).get_dNdW();
   for (unsigned int i=0; i<dNdW.rowDim(); i++)
   {
      for (unsigned int j=0; j<dNdW.colDim(); j++)
      {
         cout << dNdW.at(i,j) << " ";
      }
      cout << endl;
   }
   cout << "-----------------------" << endl;

   cout << "----- actor out input err -----" << endl;
   const vector<double>& inerr = actor->layer(oindx-1).get_input_error();
   for (unsigned int i=0; i<inerr.size(); i++)
      cout << inerr.at(i) << " ";
   cout << endl;
   cout << "-----------------------" << endl;
   */
}

} /* namespace flex_neuralnet */
