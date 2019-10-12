//
// Created by kfedrick on 8/25/19.
//

#include "NetworkLayer.h"

using flexnnet::NetworkLayer;

NetworkLayer::NetworkLayer(size_t _sz, const std::string& _name, NetworkLayerType _type) : BasicLayer(_sz, _name, _type)
{}

NetworkLayer::~NetworkLayer()
{
}
