/*
 * DeltaBarDeltaNetworkLearningRate.h
 *
 *  Created on: May 3, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_DELTA_BAR_DELTA_LRATE_POLICY_H_
#define FLEX_NEURALNET_DELTA_BAR_DELTA_LRATE_POLICY_H_

#include "Array.h"
#include "LearningRatePolicy.h"

#include <iostream>
#include <vector>
#include <map>

using namespace std;

namespace flex_neuralnet
{

/*
 * Learning rate policy for a layer. Responsible for providing the
 * learning rates for the biases and layer weights for a layer, and
 * for updating the learning rates according to policy. The base class
 * implements a fixed,non-adaptive learning rate policy.
 */

class DeltaBarDeltaLearningRate: public LearningRatePolicy
{
public:
   DeltaBarDeltaLearningRate();
   DeltaBarDeltaLearningRate(const BaseNeuralNet& _layer);
   DeltaBarDeltaLearningRate(
         const DeltaBarDeltaLearningRate& _nnLRPolicy);
   ~DeltaBarDeltaLearningRate();

   void set_global_learning_rate(double _rate);
   void set_smoothing_constant(double _lambda);
   void set_min_learning_rate(double _min);
   void set_max_learning_rate(double _max);
   void set_increment_value(double _inc);
   void set_reduction_factor(double _factor);

   DeltaBarDeltaLearningRate& operator=(
         const DeltaBarDeltaLearningRate& _nnLRPolicy);

   void reset();
   void update_learning_rate_adjustments(unsigned int _timeStep = 1);
   void apply_learning_rate_adjustments();
   void reduce_learning_rate(double _reductionFactor);

protected:
   void copy(const DeltaBarDeltaLearningRate& _nnLRPolicy);

private:
   void alloc_storage();
   void update_smoothed_gradients();
   void zero_cumulative_gradients();
   void zero_smoothed_gradients();
   double calc_adjusted_learning_rate(double _currRate, double _localGrad,
         double _smoothedGrad);
   double calc_reduced_rate(double _currRate, double _reductionFactor);
   ;

private:
   double min_learning_rate;
   double max_learning_rate;
   double increment_value;
   double reduction_factor;
   double smoothing_constant;

private:
   map<string, vector<double> > cumulative_dEdB;
   map<string, Array<double> > cumulative_dEdW;

   map<string, vector<double> > smoothed_dEdB;
   map<string, Array<double> > smoothed_dEdW;
};

inline
DeltaBarDeltaLearningRate::DeltaBarDeltaLearningRate() :
      LearningRatePolicy()
{
   neural_net = NULL;

   min_learning_rate = 1.0e-12;
   max_learning_rate = 0.95;
   increment_value = 0.01;
   reduction_factor = 0.2;
   smoothing_constant = 0.7;
}

inline
DeltaBarDeltaLearningRate::DeltaBarDeltaLearningRate(
      const BaseNeuralNet& _nn) :
      LearningRatePolicy(_nn)
{
   alloc_storage();

   min_learning_rate = 1.0e-12;
   max_learning_rate = 0.95;
   increment_value = 0.01;
   reduction_factor = 0.2;
   smoothing_constant = 0.7;
}

inline
DeltaBarDeltaLearningRate::DeltaBarDeltaLearningRate(
      const DeltaBarDeltaLearningRate& _nnLRPolicy) :
      LearningRatePolicy()
{
   copy(_nnLRPolicy);
}

inline
DeltaBarDeltaLearningRate::~DeltaBarDeltaLearningRate()
{
   // Do nothing
}

inline
void DeltaBarDeltaLearningRate::set_global_learning_rate(double _rate)
{
   LearningRatePolicy::set_global_learning_rate(_rate);

   zero_smoothed_gradients();
   zero_cumulative_gradients();
}

inline
void DeltaBarDeltaLearningRate::set_min_learning_rate(double _min)
{
   min_learning_rate = _min;
}

inline
void DeltaBarDeltaLearningRate::set_max_learning_rate(double _max)
{
   max_learning_rate = _max;
}

inline
void DeltaBarDeltaLearningRate::set_increment_value(double _inc)
{
   increment_value = _inc;
}

inline
void DeltaBarDeltaLearningRate::set_reduction_factor(double _factor)
{
   reduction_factor = _factor;
}

inline
void DeltaBarDeltaLearningRate::set_smoothing_constant(double _lambda)
{
   smoothing_constant = _lambda;
}

inline
DeltaBarDeltaLearningRate& DeltaBarDeltaLearningRate::operator=(
      const DeltaBarDeltaLearningRate& _nnLRPolicy)
{
   copy(_nnLRPolicy);
   return *this;
}

inline
void DeltaBarDeltaLearningRate::copy(
      const DeltaBarDeltaLearningRate& _nnLRPolicy)
{
   /*
    * First call the base class copy function
    */
   LearningRatePolicy::copy(_nnLRPolicy);

   min_learning_rate = _nnLRPolicy.min_learning_rate;
   max_learning_rate = _nnLRPolicy.max_learning_rate;
   increment_value = _nnLRPolicy.increment_value;
   reduction_factor = _nnLRPolicy.reduction_factor;
   smoothing_constant = _nnLRPolicy.smoothing_constant;

   cumulative_dEdB = _nnLRPolicy.cumulative_dEdB;
   smoothed_dEdB = _nnLRPolicy.smoothed_dEdB;

   cumulative_dEdW = _nnLRPolicy.cumulative_dEdW;
   smoothed_dEdW = _nnLRPolicy.smoothed_dEdW;
}

inline
void DeltaBarDeltaLearningRate::alloc_storage()
{
   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();

   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      cumulative_dEdB[name].resize(layer.size());
      smoothed_dEdB[name].resize(layer.size());

      const Array<double>& layer_weights = layer.get_weights();
      cumulative_dEdW[name].resize(layer_weights.rowDim(),
            layer_weights.colDim());
      smoothed_dEdW[name].resize(layer_weights.rowDim(),
            layer_weights.colDim());
   }

   zero_smoothed_gradients();
   zero_cumulative_gradients();
}

inline
void DeltaBarDeltaLearningRate::reset()
{
   zero_cumulative_gradients();
   zero_smoothed_gradients();
}

inline
void DeltaBarDeltaLearningRate::update_learning_rate_adjustments(
      unsigned int _timeStep)
{
   /*
    * Accumulate the error gradients so we can use them later to calculate
    * the learning rate adjustments.
    */
   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();

   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      const vector<double>& dEdB = layer.get_dEdB(_timeStep);
      for (unsigned int ndx = 0; ndx < layer.size(); ndx++)
         cumulative_dEdB[name][ndx] += dEdB[ndx];

      const Array<double>& dEdW = layer.get_dEdW(_timeStep);
      cumulative_dEdW[name] += dEdW;
   }
}

inline
void DeltaBarDeltaLearningRate::apply_learning_rate_adjustments()
{
   double curr_rate;
   double local_gradient;
   double smoothed_gradient;

   update_smoothed_gradients();

   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();

   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      vector<double>& layer_bias_learning_rates =
            layer_bias_learning_rates_map[name];
      for (unsigned int ndx = 0; ndx < layer_bias_learning_rates.size(); ndx++)
      {
         curr_rate = layer_bias_learning_rates[ndx];
         local_gradient = cumulative_dEdB[name][ndx];
         smoothed_gradient = smoothed_dEdB[name][ndx];

         layer_bias_learning_rates[ndx] = calc_adjusted_learning_rate(curr_rate,
               local_gradient, smoothed_gradient);
      }

      Array<double>& layer_weight_learning_rates =
            layer_weight_learning_rates_map[name];
      for (unsigned int to = 0; to < layer_weight_learning_rates.rowDim(); to++)
      {
         for (unsigned int from = 0;
               from < layer_weight_learning_rates.colDim(); from++)
         {
            curr_rate = layer_weight_learning_rates.at(to, from);
            local_gradient = cumulative_dEdW[name].at(to, from);
            smoothed_gradient = smoothed_dEdW[name].at(to, from);

            layer_weight_learning_rates.at(to, from) =
                  calc_adjusted_learning_rate(curr_rate, local_gradient,
                        smoothed_gradient);
         }
      }
   }

   zero_cumulative_gradients();
}

inline
void DeltaBarDeltaLearningRate::reduce_learning_rate(
      double _reductionFactor)
{
   double curr_rate;

   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();

   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      vector<double>& layer_bias_learning_rates =
            layer_bias_learning_rates_map[name];
      for (unsigned int ndx = 0; ndx < layer_bias_learning_rates.size(); ndx++)
      {
         curr_rate = layer_bias_learning_rates[ndx];
         layer_bias_learning_rates[ndx] = calc_reduced_rate(curr_rate, _reductionFactor);
      }

      Array<double>& layer_weight_learning_rates =
            layer_weight_learning_rates_map[name];
      for (unsigned int to = 0; to < layer_weight_learning_rates.rowDim(); to++)
      {
         for (unsigned int from = 0;
               from < layer_weight_learning_rates.colDim(); from++)
         {
            curr_rate = layer_weight_learning_rates.at(to, from);
            layer_weight_learning_rates.at(to, from) = calc_reduced_rate(curr_rate, _reductionFactor);
         }
      }
   }
}

inline
double DeltaBarDeltaLearningRate::calc_reduced_rate(double _currRate,
      double _reductionFactor)
{
   double new_rate = 0;

   new_rate = _reductionFactor * _currRate;
   if (_currRate != 0)
      new_rate = (new_rate > min_learning_rate) ? new_rate : min_learning_rate;

   return new_rate;
}

inline
void DeltaBarDeltaLearningRate::update_smoothed_gradients()
{
   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();

   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      vector<double>& cumulative_layer_dEdB = cumulative_dEdB[name];
      vector<double>& smoothed_layer_dEdB = smoothed_dEdB[name];
      for (unsigned int ndx = 0; ndx < smoothed_layer_dEdB.size(); ndx++)
         smoothed_layer_dEdB[ndx] = smoothing_constant * smoothed_layer_dEdB[ndx]
               + (1 - smoothing_constant) * cumulative_layer_dEdB[ndx];

      Array<double>& cumulative_layer_dEdW = cumulative_dEdW[name];
      Array<double>& smoothed_layer_dEdW = smoothed_dEdW[name];
      for (unsigned int out_ndx = 0; out_ndx < smoothed_layer_dEdW.rowDim(); out_ndx++)
      {
         for (unsigned int in_ndx = 0; in_ndx < smoothed_layer_dEdW.colDim(); in_ndx++)
            smoothed_layer_dEdW.at(out_ndx, in_ndx) = smoothing_constant
                  * smoothed_layer_dEdW.at(out_ndx, in_ndx)
                  + (1 - smoothing_constant) * cumulative_layer_dEdW.at(out_ndx, in_ndx);
      }
   }
}

inline
void DeltaBarDeltaLearningRate::zero_cumulative_gradients()
{
   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();

   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      cumulative_dEdB[name].assign(cumulative_dEdB[name].size(), 0.0);
      cumulative_dEdW[name] = 0.0;
   }
}

inline
void DeltaBarDeltaLearningRate::zero_smoothed_gradients()
{
   const vector<BaseLayer*> network_layers = neural_net->get_network_layers();

   for (unsigned int ndx = 0; ndx < network_layers.size(); ndx++)
   {
      BaseLayer& layer = *network_layers[ndx];
      string name = layer.name();

      smoothed_dEdB[name].assign(smoothed_dEdB[name].size(), 0.0);
      smoothed_dEdW[name] = 0.0;
   }
}

inline
double DeltaBarDeltaLearningRate::calc_adjusted_learning_rate(
      double _currRate, double _localGrad, double _smoothedGrad)
{
   double new_rate;

   /*
    * If the current learning rate is zero then never modify it.
    */
   if (_currRate == 0)
   {
      return 0.0;
   }
   else
   {
      double temp = _localGrad * _smoothedGrad;

      if (temp > 0)
         new_rate = _currRate + increment_value;
      else if (temp < 0)
         new_rate = reduction_factor * _currRate;
      else
         new_rate = _currRate;

      if (new_rate > max_learning_rate)
         new_rate = max_learning_rate;
      else if (new_rate < min_learning_rate)
         new_rate = min_learning_rate;

      return new_rate;
   }
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_DELTA_BAR_DELTA_LRATE_POLICY_H_ */
