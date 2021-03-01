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
      const Array2D<double>& const_dAdN_ref = dAdN;
      const Array2D<double>& const_dNdW_ref = dNdW;
      const Array2D<double>& const_dNdI_ref = dNdI;

      bool stale_dAdN;
      bool stale_dNdW;
      bool stale_dNdI;

      Array2D<double> dAdN;
      Array2D<double> dNdW;
      Array2D<double> dNdI;

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
      stale_dAdN = true;
      stale_dNdW = true;
      stale_dNdI = true;
   }
}

#endif //FLEX_NEURALNET_LAYERDERIVATIVES_H_
