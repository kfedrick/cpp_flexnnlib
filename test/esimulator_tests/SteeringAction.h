//
// Created by kfedrick on 5/8/21.
//

#ifndef _STEERINGACTION_H_
#define _STEERINGACTION_H_

#include <ActionView.h>
#include "DerbySim.h"

enum class ActionEnum { Left, Right };

class SteeringAction : public flexnnet::ActionView<ActionEnum>
{
   ActionEnum decode_action(void)
   {
      return ActionEnum::Right;
   }

};

#endif //_STEERINGACTION_H_
