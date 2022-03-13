//
// Created by kfedrick on 5/8/21.
//

#ifndef _STEERINGACTION_H_
#define _STEERINGACTION_H_

#include <FeatureSetImpl.h>
#include "SteeringActionFeature.h"

//enum class ActionEnum { Left, Right };

class SteeringAction : public flexnnet::FeatureSetImpl<std::tuple<SteeringActionFeature>>
{
public:
   SteeringAction();
   SteeringAction(const FeatureSetImpl<std::tuple<SteeringActionFeature>>& _ta);
   SteeringAction& operator=(const FeatureSetImpl<std::tuple<SteeringActionFeature>>& _ta);

   SteeringActionFeature::ActionEnum get_action(void) const
   {
      return std::get<0>(get_features()).get_action();
   }
};

SteeringAction::SteeringAction() : FeatureSetImpl<std::tuple<SteeringActionFeature>>()
{
}

SteeringAction::SteeringAction(const FeatureSetImpl<std::tuple<SteeringActionFeature>>& _ta) : FeatureSetImpl<std::tuple<SteeringActionFeature>>(_ta)
{
   std::cout << "TestAction::copy constructor\n";
   std::cout << "value<0>[0] " << std::get<0>(this->get_features()).get_encoding()[0] << "\n";

   *this = _ta;
}

SteeringAction& SteeringAction::operator=(const FeatureSetImpl<std::tuple<SteeringActionFeature>>& _ta)
{
   std::cout << "TestAction::operator=\n";
   std::cout << "value<0>[0] " << std::get<0>(this->get_features()).get_encoding()[0] << "\n";

   *this = _ta;
   return *this;
}

#endif //_STEERINGACTION_H_
