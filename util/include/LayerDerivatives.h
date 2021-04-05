//
// Created by kfedrick on 5/12/19.
//

#ifndef FLEX_NEURALNET_LAYERDERIVATIVES_H_
#define FLEX_NEURALNET_LAYERDERIVATIVES_H_

#include "Array2D.h"

namespace flexnnet
{
   class LayerDerivatives
   {
   public:
      LayerDerivatives();
      LayerDerivatives(size_t _out_sz, size_t _rawin_sz);
      LayerDerivatives(const LayerDerivatives& _lderiv);
      LayerDerivatives(LayerDerivatives&& _lderiv);
      ~LayerDerivatives();

      void resize(size_t _out_sz, size_t _rawin_sz);
      void stale();

      LayerDerivatives& operator=(const LayerDerivatives& _lderiv);
      LayerDerivatives& operator=(LayerDerivatives&& _lderiv);

   private:
      void initialize();
      void copy(const LayerDerivatives& _weights);
      void copy(LayerDerivatives&& _weights);

   public:
      const Array2D<double>& const_dy_dnet_ref = dy_dnet;
      const Array2D<double>& const_dnet_dw_ref = dnet_dw;
      const Array2D<double>& const_dnet_dx_ref = dnet_dx;
      const Array2D<double>& const_dE_dw_ref = dE_dw;

      bool stale_dy_dnet;
      bool stale_dnet_dw;
      bool stale_dnet_dx;
      bool stale_dE_dw;

      Array2D<double> dy_dnet;
      Array2D<double> dnet_dw;
      Array2D<double> dnet_dx;
      Array2D<double> dE_dw;
   };

   inline LayerDerivatives::LayerDerivatives(size_t _out_sz, size_t _rawin_sz)
   {
      resize(_out_sz, _rawin_sz);
      initialize();
   }

   inline void LayerDerivatives::initialize()
   {
      stale();
   }

   /**
    * Mark all derivative arrays as stale (e.g. after new activation).
    */
   inline void LayerDerivatives::stale()
   {
      stale_dy_dnet = true;
      stale_dnet_dw = true;
      stale_dnet_dx = true;
      stale_dE_dw = true;
   }
}

#endif //FLEX_NEURALNET_LAYERDERIVATIVES_H_
