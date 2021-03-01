//
// Created by kfedrick on 2/21/21.
//

#include "NetworkLayerImpl.h"

#include <iostream>
#include "flexnnet_networks.h"

using flexnnet::BasicLayer;
using flexnnet::NetworkLayerImpl;
using flexnnet::LayerConnRecord;

NetworkLayerImpl::NetworkLayerImpl() : NetworkLayer()
{
}

NetworkLayerImpl::NetworkLayerImpl(bool _is_output) : NetworkLayer(_is_output)
{
}

NetworkLayerImpl::NetworkLayerImpl(const std::shared_ptr<BasicLayer>& _layer, bool _is_output) : NetworkLayer(_layer, _is_output)
{
}

NetworkLayerImpl::NetworkLayerImpl(std::shared_ptr<BasicLayer>&& _layer, bool _is_output) : NetworkLayer(std::forward<std::shared_ptr<BasicLayer>>(_layer), _is_output)
{
}

NetworkLayerImpl::~NetworkLayerImpl() {}

const std::valarray<double>& NetworkLayerImpl::activate(const NNetIO_Typ& _externin)
{
   const std::valarray<double>& _externin_vec = marshal_inputs(_externin);
   return basiclayer()->activate(_externin_vec);
}
