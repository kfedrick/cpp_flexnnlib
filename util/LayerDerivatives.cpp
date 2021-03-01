//
// Created by kfedrick on 5/12/19.
//

#include "LayerDerivatives.h"

using flexnnet::LayerDerivatives;

LayerDerivatives::LayerDerivatives()
{
   initialize();
}

LayerDerivatives::~LayerDerivatives() {}

void LayerDerivatives::resize(size_t _out_sz, size_t _rawin_sz)
{
   dAdN.resize(_out_sz, _out_sz);
   dNdW.resize(_out_sz, _rawin_sz + 1);
   dNdI.resize(_out_sz, _rawin_sz + 1);
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
   dAdN.set(_lderiv.dAdN);
   dNdI.set(_lderiv.dNdI);
   dNdW.set(_lderiv.dNdW);

   stale_dAdN = _lderiv.stale_dAdN;
   stale_dNdW = _lderiv.stale_dNdW;
   stale_dNdI = _lderiv.stale_dNdI;
}

void LayerDerivatives::copy(LayerDerivatives&& _lderiv)
{
   dAdN.set(std::forward<const Array2D<double>>(_lderiv.dAdN));
   dNdI.set(std::forward<const Array2D<double>>(_lderiv.dNdI));
   dNdW.set(std::forward<const Array2D<double>>(_lderiv.dNdW));

   stale_dAdN = _lderiv.stale_dAdN;
   stale_dNdW = _lderiv.stale_dNdW;
   stale_dNdI = _lderiv.stale_dNdI;
}
