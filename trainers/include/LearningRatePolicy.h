/*
 * NetworkLayerLearningRate.h
 *
 *  Created on: May 3, 2015
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_NEWLRATE_POLICY_H_
#define FLEX_NEURALNET_NEWLRATE_POLICY_H_

#include "Array2D.h"

#include <cstddef>
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

   class LearningRatePolicy
   {
   public:
      static constexpr double DEFAULT_LEARNING_RATE{0.01};
      static constexpr double DEFAULT_REDUCTION_FACT{0.4};

   public:
      LearningRatePolicy();
      LearningRatePolicy(const BaseNeuralNet& _nnet);
      LearningRatePolicy(const LearningRatePolicy& _nnLRPolicy);
      virtual ~LearningRatePolicy();

      void
      initialize(const BaseNeuralNet& _nnet);

      const Array2D<double>&
      get_learning_rates(const std::string& _layerID) const;

      const std::map<std::string, Array2D<double> >&
      get_learning_rates() const;

      virtual void
      set_init_learning_rate(double _rate);

      virtual void
      init_learning_rate();

      virtual void
      set_learning_rate(const std::string& _layerID, double _rate);

      LearningRatePolicy&
      operator=(const LearningRatePolicy& _nnLRPolicy);

      virtual void
      clear_learning_rate_adjustments();

      virtual void
      calc_learning_rate_adjustment(const BaseNeuralNet& _nnet, unsigned int _timeStep = 1);

      virtual void
      apply_learning_rate_adjustments();

      virtual void
      reduce_learning_rate(double _reductionFactor = DEFAULT_REDUCTION_FACT);

   protected:
      void
      copy(const LearningRatePolicy& _nnLR);

   private:
      void
      alloc_storage(const BaseNeuralNet& _nnet);

   protected:
      std::map<std::string, Array2D<double>> layer_weight_learning_rates_map;

   private:
      double initial_learning_rate;
   };

   inline LearningRatePolicy::LearningRatePolicy()
   {
      initial_learning_rate = DEFAULT_LEARNING_RATE;
   }

   inline LearningRatePolicy::LearningRatePolicy(const
                                                 BaseNeuralNet& _nnet)
   {
      alloc_storage(_nnet);
   }

   inline
   LearningRatePolicy::LearningRatePolicy(
      const LearningRatePolicy& _nnLRPolicy)
   {
      copy(_nnLRPolicy);
   }

   inline LearningRatePolicy::~LearningRatePolicy()
   {
   }

   inline
   void
   LearningRatePolicy::initialize(const BaseNeuralNet& _nnet)
   {
      alloc_storage(_nnet);
   }

   inline
   void
   LearningRatePolicy::alloc_storage(const BaseNeuralNet& _nnet)
   {
      layer_weight_learning_rates_map.clear();

      const std::map<std::string, std::shared_ptr<NetworkLayer>>
         & network_layers = _nnet.get_layers();
      for (auto it = network_layers.begin(); it != network_layers.end(); it++)
      {
         std::string id = it->first;
         const NetworkLayer& layer = *it->second;

         const LayerWeights& layer_weights = layer.weights();
         Array2D<double>::Dimensions dim = layer_weights.size();
         layer_weight_learning_rates_map[id].resize(dim.rows, dim.cols, initial_learning_rate);
      }
   }

   inline
   void
   LearningRatePolicy::copy(
      const LearningRatePolicy& _nnLRPolicy)
   {
      layer_weight_learning_rates_map = _nnLRPolicy.layer_weight_learning_rates_map;
   }

   inline
   void
   LearningRatePolicy::set_init_learning_rate(double _rate)
   {
      if (_rate < 0.0)
         throw std::invalid_argument("Error : learning rate cannot be negative");
      initial_learning_rate = _rate;

      init_learning_rate();
   }

   inline
   void
   LearningRatePolicy::init_learning_rate()
   {
      for (auto it = layer_weight_learning_rates_map.begin();
           it != layer_weight_learning_rates_map.end(); it++)
         it->second = initial_learning_rate;
   }

   inline
   void
   LearningRatePolicy::set_learning_rate(const std::string& _layerID, double _rate)
   {
      if (layer_weight_learning_rates_map.find(_layerID)
          == layer_weight_learning_rates_map.end())
      {
         std::ostringstream err_str;
         err_str << "Error (set_layer_weights_learning_rate() : Layer ID ("
                 << _layerID.c_str() << ") not found.";
         throw std::invalid_argument(err_str.str());
      }

      layer_weight_learning_rates_map[_layerID] = _rate;
   }


   inline const std::map<std::string, Array2D<double> >&
   LearningRatePolicy::get_learning_rates() const
   {
      return layer_weight_learning_rates_map;
   }

   inline const Array2D<double>&
   LearningRatePolicy::get_learning_rates(const std::string& _layerID) const
   {
      return layer_weight_learning_rates_map.at(_layerID);
   }

   inline LearningRatePolicy&
   LearningRatePolicy::operator=(
      const LearningRatePolicy& _nnLRPolicy)
   {
      copy(_nnLRPolicy);
      return *this;
   }

   inline
   void
   LearningRatePolicy::clear_learning_rate_adjustments()
   {
      // Base class implements a fixed, non-adaptive policy. Do nothing.
   }

   inline
   void
   LearningRatePolicy::calc_learning_rate_adjustment(const BaseNeuralNet& _nnet,
      unsigned int _timeStep)
   {
      // Base class implements a fixed, non-adaptive policy. Do nothing.
   }

   inline
   void
   LearningRatePolicy::apply_learning_rate_adjustments()
   {
      // Base class implements a fixed, non-adaptive policy. Do nothing.
   }

   inline
   void
   LearningRatePolicy::reduce_learning_rate(double _reductionFactor)
   {
      if (_reductionFactor <= 0 || 1 <= _reductionFactor)
      {
         std::ostringstream err_str;
         err_str << "Error (LearningRatePolicy::reduce_learning_rate() - invalid reduction factor ("
                 << _reductionFactor << ") specified.";
         throw std::invalid_argument(err_str.str());
      }

      for (auto it = layer_weight_learning_rates_map.begin();
           it != layer_weight_learning_rates_map.end(); it++)
         it->second *= _reductionFactor;
   }

} /* namespace flexnnet */

#endif /* FLEX_NEURALNET_LRATE_POLICY_H_ */
