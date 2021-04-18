//
// Created by kfedrick on 4/7/21.
//
#include "LayerState.h"

using flexnnet::LayerState;

LayerState::LayerState()
{
}

LayerState::~LayerState() {}


LayerState::LayerState(const LayerState& _state)
{
   copy(_state);
}

LayerState::LayerState(LayerState&& _state)
{
   copy(_state);
}
