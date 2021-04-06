//
// Created by kfedrick on 6/5/19.
//
#include "OldLayerConnRecord.h"
#include "BasicLayer.h"
#include "OldNetworkLayer.h"

using flexnnet::OldLayerConnRecord;

OldLayerConnRecord::OldLayerConnRecord(BasicLayer* _from, ConnectionType _type)
   : input_layer(_from), connection_type(_type), input_layer_size(_from->size())
{
}

OldLayerConnRecord::OldLayerConnRecord(size_t _sz, ConnectionType _type)
   : input_layer_size(_sz), connection_type(_type)
{
}

OldLayerConnRecord::OldLayerConnRecord(std::string _id, size_t _sz, ConnectionType _type)
   : input_layer_id(_id), input_layer_size(_sz), connection_type(_type)
{
}