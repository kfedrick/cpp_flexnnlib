//
// Created by kfedrick on 8/25/19.
//

#include "OldNetworkLayer.h"

using flexnnet::OldNetworkLayer;

OldNetworkLayer::OldNetworkLayer(size_t _sz, const std::string& _name, NetworkLayerType _type)
   : BasicLayer(_sz, _name), network_layer_type(_type)
{}

OldNetworkLayer::~OldNetworkLayer()
{
}
