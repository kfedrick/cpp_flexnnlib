//
// Created by kfedrick on 6/5/19.
//
#include "LayerConnRecord.h"
#include "BasicLayer.h"
#include "NetworkLayer.h"

using flexnnet::LayerConnRecord;

LayerConnRecord::LayerConnRecord (BasicLayer *_from, ConnectionType _type)
   : input_layer (_from), connection_type (_type), input_layer_size (_from->size ())
{
}

LayerConnRecord::LayerConnRecord (size_t _sz, ConnectionType _type)
  : input_layer_size(_sz), connection_type(_type)
{
}

LayerConnRecord::LayerConnRecord (std::string _id, size_t _sz, ConnectionType _type)
   : input_layer_id(_id), input_layer_size(_sz), connection_type(_type)

{
}