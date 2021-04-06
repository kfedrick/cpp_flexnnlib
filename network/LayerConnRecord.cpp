//
// Created by kfedrick on 2/21/21.
//

#include "LayerConnRecord.h"

using flexnnet::LayerConnRecord;

LayerConnRecord::LayerConnRecord()
{}

LayerConnRecord::LayerConnRecord(const std::shared_ptr<NetworkLayer>& _from_layer, ConnectionType _type)
   : from_layer(_from_layer), connection_type(_type)
{
}

LayerConnRecord::~LayerConnRecord()
{}
