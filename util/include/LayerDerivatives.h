//
// Created by kfedrick on 5/12/19.
//

#ifndef FLEX_NEURALNET_LAYERDERIVATIVES_H_
#define FLEX_NEURALNET_LAYERDERIVATIVES_H_

#include "Array.h"

namespace flexnnet
{
   class LayerDerivatives
   {
   public:
      LayerDerivatives ();
      LayerDerivatives (unsigned int _out_sz, unsigned int _netin_sz, unsigned int _rawin_sz);
      ~LayerDerivatives ();

      void resize (unsigned int _out_sz, unsigned int _netin_sz, unsigned int _rawin_sz);
      void stale();

   private:
      void initialize();

   public:
      const Array<double> &const_dAdN_ref = dAdN;
      const Array<double> &const_dNdW_ref = dNdW;
      const Array<double> &const_dNdI_ref = dNdI;

      bool stale_dAdN;
      bool stale_dNdW;
      bool stale_dNdI;

      Array<double> dAdN;
      Array<double> dNdW;
      Array<double> dNdI;

   };

   inline LayerDerivatives::LayerDerivatives (unsigned int _out_sz, unsigned int _netin_sz, unsigned int _rawin_sz)
   {
      resize(_out_sz, _netin_sz, _rawin_sz);
      initialize();
   }

   inline LayerDerivatives::LayerDerivatives ()
   {
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

   inline LayerDerivatives::~LayerDerivatives ()
   {

   }

   inline void LayerDerivatives::resize (unsigned int _out_sz, unsigned int _netin_sz, unsigned int _rawin_sz)
   {
      dAdN.resize(_out_sz, _netin_sz);
      dNdW.resize(_netin_sz, _rawin_sz+1);
      dNdI.resize(_netin_sz, _rawin_sz+1);
   }

}

#endif //FLEX_NEURALNET_LAYERDERIVATIVES_H_
