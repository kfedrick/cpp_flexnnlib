//
// Created by kfedrick on 9/30/19.
//

#ifndef FLEX_NEURALNET_TDTRAINERCONFIG_H_
#define FLEX_NEURALNET_TDTRAINERCONFIG_H_

#include "TrainerConfig.h"

namespace flexnnet
{

   class TDTrainerConfig
   {
   public:
      enum TD_TRAINING_MODE { FINAL_COST, COST_TO_GO };
      static constexpr double DEFAULT_GAMMA = 0.2;
      static constexpr double DEFAULT_LAMBDA = 0.4;

   public:
      /**
       *
       * @param _mode
       */
      void set_td_mode(TD_TRAINING_MODE _mode);

      /**
       * Set the vectorize of the cumulative temporal-difference reinforcement
       * discount parameter, gamma : (0 <= get_gamma <= 1.0)
       *
       * @param _val
       */
      void set_gamma(double _val);

      /**
       * Set the vectorize of the eligibility trace discount parameter
       * lambda : (0 <= get_lambda <= 1.0)
       *
       * @param _val
       */
      void set_lambda(double _val);

      constexpr TD_TRAINING_MODE get_td_mode();

      constexpr double get_gamma();

      constexpr double get_lambda();

   private:

      TD_TRAINING_MODE td_mode{COST_TO_GO};

      // td_discount is the discount parameter for cumulative temporal-difference
      // learning error and their variants. It controls how fast older reinforcement
      // signals are discounted, with: 0 <= td_discount <= 1.0
      //
      double gamma{DEFAULT_GAMMA};

      double lambda{DEFAULT_LAMBDA};

   };

   inline
   constexpr TDTrainerConfig::TD_TRAINING_MODE TDTrainerConfig::get_td_mode()
   {
      return td_mode;
   }

   inline
   constexpr double TDTrainerConfig::get_gamma()
   {
      return gamma;
   }

   inline
   constexpr double TDTrainerConfig::get_lambda()
   {
      return lambda;
   }

   inline
   void TDTrainerConfig::set_td_mode(TD_TRAINING_MODE _mode)
   {
      td_mode = _mode;
   }

   inline
   void TDTrainerConfig::set_gamma(double _val)
   {
      if (_val < 0 || _val > 1.0)
      {
         std::ostringstream err_str;
         err_str
            << "Error : TDTrainerConfig.set_gamma() - illegal vectorize " << _val << "\n";
         throw std::invalid_argument(err_str.str());
      }

      gamma = _val;
   }

   inline
   void TDTrainerConfig::set_lambda(double _val)
   {
      if (_val < 0 || _val > 1.0)
      {
         std::ostringstream err_str;
         err_str
            << "Error : TDTrainerConfig.set_lambda() - illegal vectorize " << _val << "\n";
         throw std::invalid_argument(err_str.str());
      }

      lambda = _val;
   }
}

#endif //FLEX_NEURALNET_TDTRAINERCONFIG_H_
