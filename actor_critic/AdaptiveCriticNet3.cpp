/*
 * AdaptiveCriticNet3.cpp
 *
 *  Created on: Mar 24, 2015
 *      Author: kfedrick
 */

#include <AdaptiveCriticNet3.h>
#include <NetSum.h>
#include <PureLin.h>

using namespace std;

namespace flexnnet
{
   AdaptiveCriticNet3::AdaptiveCriticNet3 () : TDCNeuralNet ("adaptive-critic")
   {
      /*
      BaseLayer<NetSum, PureLin>& outputlayer =
            this->new_output_layer<NetSum, PureLin>(1, "adaptive-critic-output");
            */
   }

   AdaptiveCriticNet3::~AdaptiveCriticNet3 ()
   {

   }

   double AdaptiveCriticNet3::get_reinforcement (const Pattern &_stateVec,
                                                 const Pattern &_actionVec, unsigned int _recurStep)
   {
      Pattern opatt;
      static vector<vector<double> > invec;
      static Pattern ipatt;

      TDCNeuralNet &net = *dynamic_cast<TDCNeuralNet *>(this);

      invec.clear ();
      invec.push_back (_stateVec);
      invec.push_back (_actionVec);
      ipatt = invec;

      /*
      cout << "invec size = " << invec.size() << endl;
      for ( int i=0; i<invec.size(); i++)
         cout << "invec(" << i << ") size " << invec.at(i).size() << endl;

      cout << "invec(1)(0) = " << invec.at(1).at(0) << endl;
      */

      //cout << "ipatt size = " << ipatt.size() << endl;


      opatt = net (ipatt, _recurStep);

      return opatt ().at (0);
   }

} /* namespace flexnnet */

