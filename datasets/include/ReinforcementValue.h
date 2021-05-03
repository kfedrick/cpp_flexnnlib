//
// Created by kfedrick on 4/22/21.
//

#ifndef FLEX_NEURALNET_REINFORCEMENTVALUE_H_
#define FLEX_NEURALNET_REINFORCEMENTVALUE_H_

#include <string>
#include <map>
#include <valarray>
#include <flexnnet.h>
#include <NNetIOInterface.h>

namespace flexnnet
{
   class ReinforcementValue : public NNetIOInterface
   {
   public:
      ReinforcementValue();
      ReinforcementValue(double _rval);

      size_t size(void) const;

      const std::valarray<double>& value(void) const override;

      const flexnnet::ValarrMap& value_map(void) const override;

      void parse(const flexnnet::ValarrMap& _vmap) override;

   private:
      ValarrMap reinforcement_value_map;
   };

   ReinforcementValue::ReinforcementValue()
   {
      reinforcement_value_map["value"] = {0};
   }

   ReinforcementValue::ReinforcementValue(double _rval)
   {
      reinforcement_value_map["value"] = {_rval};

   }

   inline
   size_t ReinforcementValue::size(void) const
   {
      return 1;
   }

   const std::valarray<double>& ReinforcementValue::value(void) const
   {
      return reinforcement_value_map.at("value");
   }

   const ValarrMap& ReinforcementValue::value_map(void) const
   {
      return reinforcement_value_map;
   }

   void ReinforcementValue::parse(const flexnnet::ValarrMap& _vmap)
   {
      // TODO - fix this, validate _vmap form and contents.
      reinforcement_value_map = _vmap;
   }
}

#endif //_REINFORCEMENTVALUE_H_
