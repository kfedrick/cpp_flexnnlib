/*
 * TDNeuralNEt.h
 *
 *  Created on: Jan 14, 2018
 *      Author: kfedrick
 */

#ifndef TDCNEURALNET_H_
#define TDCNEURALNET_H_

#include <vector>

#include "Array.h"
#include "BaseNeuralNet.h"
#include "HvMap.h"

using namespace std;

namespace flex_neuralnet
{

class TDCNeuralNet : public flex_neuralnet::BaseNeuralNet
{
public:
   TDCNeuralNet(const char* _name  = "BaseNeuralNet");
   TDCNeuralNet(const string& _name);
   virtual ~TDCNeuralNet();

   void resize_history(unsigned int sz = 2);
   void clear_hv();

   virtual const Pattern& operator()(const Pattern& ipattern, unsigned int recurStep = 1);
   virtual const PatternSequence& operator()(const PatternSequence& ipattern, unsigned int recurStep = 1);

   virtual void backprop(const vector<double>& isse, unsigned int timeStep = 1);
   virtual void backprop_scatter(BaseLayer& layer, unsigned int timeStep = 1, unsigned int closedLoopStep = 0);

   void set_v(const string& _name, const Array<double>& _v);
   const Array<double>& get_Hv(const string& _name);

   virtual void clear_error(unsigned int timeStep = 1);


private:

   void init_tdcnet();

private:
   bool hv_initialized_flag;

   map<const BaseLayer*, HvMap* > hv_conn_map;

   map< string, vector<double> > Ry_map;
   map< string, vector<double> > Rx_map;

   map< string, vector<double> > RdEdy_map;
   map< string, vector<double> > RdEdx_map;

   map< string, Array<double> > v_map;
   map< string, Array<double> > Hv_map;
};

} /* namespace flex_neuralnet */

#endif /* TDCNEURALNET_H_ */
