//
// Created by kfedrick on 9/30/19.
//

#ifndef FLEX_NEURALNET_TDEVALUATORCONFIG_H_
#define FLEX_NEURALNET_TDEVALUATORCONFIG_H_

#include "BasicEvalConfig.h"
#include "TDTrainerConfig.h"

namespace flexnnet
{
   template<TDForecastMode _MODE>
   class TDEvaluatorConfig : public BasicEvalConfig
   {
   public:
      static constexpr double DEFAULT_TD_DISCOUNT = 0.2;
      static const TDForecastMode td_forecast_mode{_MODE};

   public:
      /**
       * Set the value of the cumulative temporal-difference error discount
       * discount parameter, gamma : (0 <= gamma <= 1.0)
       *
       * @param _gamma
       */
      void set_td_discount(double _gamma);

      /**
       * Get the value of the cumulative temporal-difference error discount
       * paramater, gamma.
       *
       * @return
       */
      constexpr double td_discount(void);

   private:
      // td_discount is the discount parameter for cumulative temporal-difference
      // learning error and their variants. It controls how fast older reinforcement
      // signals are discounted, with: 0 <= td_discount <= 1.0
      //
      double gamma{DEFAULT_TD_DISCOUNT};
   };

   template<TDForecastMode _MODE>
   inline void TDEvaluatorConfig<_MODE>::set_td_discount(double _gamma)
   {
      if (_gamma < 0 || _gamma > 1.0)
      {
         std::ostringstream err_str;
         err_str
            << "Error : TDEvaluatorConfig.set_td_discount() - illegal value " << _gamma << "\n";
         throw std::invalid_argument(err_str.str());
      }

      gamma = _gamma;
   }

   template<TDForecastMode _MODE>
   inline constexpr double TDEvaluatorConfig<_MODE>::td_discount(void)
   {
      return gamma;
   }
}

#endif //FLEX_NEURALNET_TDEVALUATORCONFIG_H_
