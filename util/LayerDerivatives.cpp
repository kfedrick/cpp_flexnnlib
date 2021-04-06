//
// Created by kfedrick on 5/12/19.
//

#include <iostream>
#include "LayerDerivatives.h"

using flexnnet::LayerDerivatives;

LayerDerivatives::LayerDerivatives()
{
   initialize();
}

LayerDerivatives::~LayerDerivatives() {}

void LayerDerivatives::resize(size_t _out_sz, size_t _rawin_sz)
{
   dy_dnet.resize(_out_sz, _out_sz);
   dnet_dw.resize(_out_sz, _rawin_sz + 1);
   dE_dw.resize(_out_sz, _rawin_sz + 1);
   dnet_dx.resize(_out_sz, _rawin_sz);
}

LayerDerivatives::LayerDerivatives(const LayerDerivatives& _lderiv)
{
   copy(_lderiv);
}

LayerDerivatives::LayerDerivatives(LayerDerivatives&& _lderiv)
{
   copy(_lderiv);
}

LayerDerivatives& LayerDerivatives::operator=(const LayerDerivatives& _lderiv)
{
   copy(_lderiv);
   return *this;
}

LayerDerivatives& LayerDerivatives::operator=(LayerDerivatives&& _lderiv)
{
   copy(_lderiv);
   return *this;
}

void LayerDerivatives::copy(const LayerDerivatives& _lderiv)
{
   dy_dnet.set(_lderiv.dy_dnet);
   dnet_dx.set(_lderiv.dnet_dx);
   dnet_dw.set(_lderiv.dnet_dw);
   dE_dw.set(_lderiv.dE_dw);

   stale_dy_dnet = _lderiv.stale_dy_dnet;
   stale_dnet_dw = _lderiv.stale_dnet_dw;
   stale_dnet_dx = _lderiv.stale_dnet_dx;
   stale_dE_dw = _lderiv.stale_dE_dw;
}

void LayerDerivatives::copy(LayerDerivatives&& _lderiv)
{
   dy_dnet.set(std::forward<const Array2D<double>>(_lderiv.dy_dnet));
   dnet_dx.set(std::forward<const Array2D<double>>(_lderiv.dnet_dx));
   dnet_dw.set(std::forward<const Array2D<double>>(_lderiv.dnet_dw));
   dE_dw.set(std::forward<const Array2D<double>>(_lderiv.dE_dw));

   stale_dy_dnet = _lderiv.stale_dy_dnet;
   stale_dnet_dw = _lderiv.stale_dnet_dw;
   stale_dnet_dx = _lderiv.stale_dnet_dx;
   stale_dE_dw = _lderiv.stale_dE_dw;
}
