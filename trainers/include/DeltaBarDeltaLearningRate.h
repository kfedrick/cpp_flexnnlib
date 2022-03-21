/*
 * DeltaBarDeltaNetworkLearningRate.h
 *
 *  Created on: May 3, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_DELTA_BAR_DELTA_LRATE_POLICY_H_
#define FLEX_NEURALNET_DELTA_BAR_DELTA_LRATE_POLICY_H_

#include "Array2D.h"
#include "LearningRatePolicy.h"

#include <iostream>
#include <vector>
#include <map>

namespace flexnnet
{

/*
 * Learning rate policy for a layer. Responsible for providing the
 * learning rates for the biases and layer weights for a layer, and
 * for updating the learning rates according to policy. The base class
 * implements a fixed,non-adaptive learning rate policy.
 */

   class DeltaBarDeltaLearningRate : public LearningRatePolicy
   {
   public:
      DeltaBarDeltaLearningRate();
      DeltaBarDeltaLearningRate(const BaseNeuralNet& _nnet);
      DeltaBarDeltaLearningRate(
         const DeltaBarDeltaLearningRate& _nnLRPolicy);
      ~DeltaBarDeltaLearningRate();

      void
      set_init_learning_rate(double _rate) override;

      void
      init_learning_rate() override;

      void
      set_smoothing_constant(double _lambda);

      void
      set_min_learning_rate(double _min);

      void
      set_max_learning_rate(double _max);

      void
      set_increment_value(double _inc);

      void
      set_reduction_factor(double _factor);

      DeltaBarDeltaLearningRate&
      operator=(
         const DeltaBarDeltaLearningRate& _nnLRPolicy);

      void
      clear_learning_rate_adjustments() override;

      void
      calc_learning_rate_adjustment(const BaseNeuralNet& _nnet, unsigned int _timeStep = 1) override;

      void
      apply_learning_rate_adjustments() override;

      void
      reduce_learning_rate(double _reductionFactor = 0.4);

   protected:
      void
      copy(const DeltaBarDeltaLearningRate& _nnLRPolicy);

   private:
      void
      alloc_storage(const BaseNeuralNet& _nnet);

      void
      update_smoothed_gradients();

      void
      zero_cumulative_gradients();

      void
      zero_smoothed_gradients();

      double
      calc_adjusted_learning_rate(double _currRate, double _localGrad,
                                  double _smoothedGrad);

      double
      calc_reduced_rate(double _currRate, double _reductionFactor);;

   private:
      double min_learning_rate{1.0e-12};
      double max_learning_rate{0.95};
      double increment_value{0.01};
      double reduction_factor{DEFAULT_REDUCTION_FACT};
      double smoothing_constant{0.7};
      double initial_learning_rate{DEFAULT_LEARNING_RATE};

   private:
      std::map<std::string, Array2D<double> > cumulative_dE_dw;
      std::map<std::string, Array2D<double> > smoothed_dE_dw;
   };

   inline
   DeltaBarDeltaLearningRate::DeltaBarDeltaLearningRate() :
      LearningRatePolicy()
   {
   }

   inline
   DeltaBarDeltaLearningRate::DeltaBarDeltaLearningRate(
      const BaseNeuralNet& _nnet) :
      LearningRatePolicy(_nnet)
   {
      alloc_storage(_nnet);
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
   void
   DeltaBarDeltaLearningRate::set_init_learning_rate(double _rate)
   {
      initial_learning_rate = _rate;
      LearningRatePolicy::set_init_learning_rate(_rate);

      zero_smoothed_gradients();
      zero_cumulative_gradients();
   }

   inline
   void
   DeltaBarDeltaLearningRate::init_learning_rate()
   {
      LearningRatePolicy::init_learning_rate();

      zero_smoothed_gradients();
      zero_cumulative_gradients();
   }

   inline
   void
   DeltaBarDeltaLearningRate::set_min_learning_rate(double _min)
   {
      min_learning_rate = _min;
   }

   inline
   void
   DeltaBarDeltaLearningRate::set_max_learning_rate(double _max)
   {
      max_learning_rate = _max;
   }

   inline
   void
   DeltaBarDeltaLearningRate::set_increment_value(double _inc)
   {
      increment_value = _inc;
   }

   inline
   void
   DeltaBarDeltaLearningRate::set_reduction_factor(double _factor)
   {
      reduction_factor = _factor;
   }

   inline
   void
   DeltaBarDeltaLearningRate::set_smoothing_constant(double _lambda)
   {
      smoothing_constant = _lambda;
   }

   inline
   DeltaBarDeltaLearningRate&
   DeltaBarDeltaLearningRate::operator=(
      const DeltaBarDeltaLearningRate& _nnLRPolicy)
   {
      copy(_nnLRPolicy);
      return *this;
   }

   inline
   void
   DeltaBarDeltaLearningRate::copy(
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

      cumulative_dE_dw = _nnLRPolicy.cumulative_dE_dw;
      smoothed_dE_dw = _nnLRPolicy.smoothed_dE_dw;
   }

   inline
   void
   DeltaBarDeltaLearningRate::alloc_storage(const BaseNeuralNet& _nnet)
   {

      for (auto it = layer_weight_learning_rates_map.begin();
           it != layer_weight_learning_rates_map.end(); it++)
      {
         std::string id = it->first;

         Array2D<double>::Dimensions dim = it->second.size();
         cumulative_dE_dw[id].resize(dim.rows, dim.cols);
         smoothed_dE_dw[id].resize(dim.rows, dim.cols);

      }

      zero_smoothed_gradients();
      zero_cumulative_gradients();
   }

   inline
   void
   DeltaBarDeltaLearningRate::clear_learning_rate_adjustments()
   {
      set_init_learning_rate(initial_learning_rate);
      zero_cumulative_gradients();
      zero_smoothed_gradients();
   }

   inline
   void
   DeltaBarDeltaLearningRate::calc_learning_rate_adjustment(const BaseNeuralNet& _nnet,
      unsigned int _timeStep)
   {
      /*
       * Accumulate the error gradients so we can use them later to calculate
       * the learning rate adjustments.
       */
      const std::map<std::string, std::shared_ptr<NetworkLayer>> network_layers = _nnet.get_layers();
      for (auto it = cumulative_dE_dw.begin(); it != cumulative_dE_dw.end(); it++)
      {
         std::string id = it->first;
         const NetworkLayer& layer = *network_layers.at(id);

         // TODO - fix this, I need dE_dw
         const Array2D<double>& dE_dw = layer.dEdw();

         cumulative_dE_dw[id] += dE_dw;
      }
   }

   inline
   void
   DeltaBarDeltaLearningRate::apply_learning_rate_adjustments()
   {
      double curr_rate;
      double local_gradient;
      double smoothed_gradient;

      update_smoothed_gradients();

      for (auto it = layer_weight_learning_rates_map.begin();
           it != layer_weight_learning_rates_map.end(); it++)
      {
         const std::string& id = it->first;

         Array2D<double>::Dimensions dim = it->second.size();
         for (unsigned int to = 0; to < dim.rows; to++)
            for (unsigned int from = 0; from < dim.cols; from++)
            {
               curr_rate = it->second.at(to, from);
               local_gradient = cumulative_dE_dw[id].at(to, from);
               smoothed_gradient = smoothed_dE_dw[id].at(to, from);

               it->second.at(to, from) =
                  calc_adjusted_learning_rate(curr_rate, local_gradient,
                                              smoothed_gradient);
            }
      }

      zero_cumulative_gradients();
   }

   inline
   void
   DeltaBarDeltaLearningRate::reduce_learning_rate(
      double _reductionFactor)
   {
      double curr_rate;

      for (auto it = layer_weight_learning_rates_map.begin();
           it != layer_weight_learning_rates_map.end(); it++)
      {
         Array2D<double>::Dimensions dim = it->second.size();
         for (unsigned int to = 0; to < dim.rows; to++)
            for (unsigned int from = 0; from < dim.cols; from++)
            {
               curr_rate = it->second.at(to, from);
               it->second.at(to, from) =
                  calc_reduced_rate(curr_rate, _reductionFactor);
            }
      }
   }

   inline
   double
   DeltaBarDeltaLearningRate::calc_reduced_rate(double _currRate,
                                                double _reductionFactor)
   {
      double new_rate = 0;

      if (_currRate == 0)
         return 0;

      new_rate = _reductionFactor * _currRate;
      new_rate = (new_rate > min_learning_rate) ? new_rate : min_learning_rate;

      return new_rate;
   }

   inline
   void
   DeltaBarDeltaLearningRate::update_smoothed_gradients()
   {
      for (auto it = smoothed_dE_dw.begin(); it != smoothed_dE_dw.end(); it++)
      {
         std::string id = it->first;

         Array2D<double>& c_dE_dw = cumulative_dE_dw[id];
         Array2D<double>::Dimensions dim = it->second.size();

         for (unsigned int to = 0; to < dim.rows; to++)
            for (unsigned int from = 0; from < dim.cols; from++)
               it->second.at(to, from) = smoothing_constant
                                             * it->second.at(to, from)
                                             + (1 - smoothing_constant)
                                                      * c_dE_dw.at(to, from);
      }
   }

   inline
   void
   DeltaBarDeltaLearningRate::zero_cumulative_gradients()
   {
      for (auto it = cumulative_dE_dw.begin(); it != cumulative_dE_dw.end(); it++)
         it->second = 0.0;
   }

   inline
   void
   DeltaBarDeltaLearningRate::zero_smoothed_gradients()
   {
      for (auto it = smoothed_dE_dw.begin(); it != smoothed_dE_dw.end(); it++)
         it->second = 0.0;
   }

   inline
   double
   DeltaBarDeltaLearningRate::calc_adjusted_learning_rate(
      double _currRate, double _localGrad, double _smoothedGrad)
   {
      double new_rate;

      /*
       * If the current learning rate is zero then never modify it.
       */
      if (_currRate == 0)
         return 0.0;

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

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_DELTA_BAR_DELTA_LRATE_POLICY_H_ */
