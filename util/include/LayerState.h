//
// Created by kfedrick on 5/13/19.
//

#ifndef FLEX_NEURALNET_LAYERSTATE_H_
#define FLEX_NEURALNET_LAYERSTATE_H_

#include "Array2D.h"

namespace flexnnet
{
   class LayerState
   {
   public:
      LayerState();
      LayerState(size_t _out_sz, size_t _rawin_sz);
      LayerState(const LayerState& _state);
      //LayerState(LayerState&& _state);
      ~LayerState();

      void resize(size_t _out_sz, size_t _rawin_sz);
      
      LayerState& operator=(const LayerState& _state);
      //LayerState& operator=(LayerState&& _state);
      
   public:
      // The cached value of the most recent raw input value.
      std::valarray<double> rawinv;

      // The net input value (e.g. net sum of the layer input)
      std::valarray<double> netinv;

      // The layer output value.
      std::valarray<double> outputv;

      // Partial derivative of layer state wrt net input.
      Array2D<double> dy_dnet;

      // Partial derivative of net input wrt layer weights.
      Array2D<double> dnet_dw;

      // Partial derivative of net input wrt layer input vector.
      Array2D<double> dnet_dx;

      // The cached value of the most recent external layer error.
      std::valarray<double> dE_dy;

      // Partial derivative of the external layer error with respect
      // to the net input value (e.g. net sum).
      std::valarray<double> dE_dnet;

      // Partial derivative of instantaneous error wrt weights.
      Array2D<double> dE_dw;

      // Partial derivative of the external layer error with respect
      // to the raw input.
      std::valarray<double> dE_dx;

   private:
      void copy(const LayerState& _state);
      //void copy(LayerState&& _state);
   };

   inline
   void LayerState::resize(size_t _out_sz, size_t _rawin_sz)
   {
      outputv.resize(_out_sz);
      netinv.resize(_out_sz);
      rawinv.resize(_rawin_sz);

      dy_dnet.resize(_out_sz, _out_sz);
      dnet_dw.resize(_out_sz, _rawin_sz + 1);
      dE_dw.resize(_out_sz, _rawin_sz + 1);
      dnet_dx.resize(_out_sz, _rawin_sz);

      dE_dy.resize(_out_sz);
      dE_dnet.resize(_out_sz);
      dE_dx.resize(_rawin_sz);
   }

   inline
   LayerState& LayerState::operator=(const LayerState& _state)
   {
      copy(_state);
      return *this;
   }

/*   inline
   LayerState& LayerState::operator=(LayerState&& _state)
   {
      copy(_state);
      return *this;
   }*/

   inline
   void LayerState::copy(const LayerState& _state)
   {
      outputv = _state.outputv;
      netinv = _state.netinv;
      rawinv = _state.rawinv;

      dy_dnet.set(_state.dy_dnet);
      dnet_dx.set(_state.dnet_dx);
      dnet_dw.set(_state.dnet_dw);
      dE_dw.set(_state.dE_dw);

      dE_dy = _state.dE_dy;
      dE_dnet = _state.dE_dnet;
      dE_dx = _state.dE_dx;
   }

/*   inline
   void LayerState::copy(LayerState&& _state)
   {
      outputv = std::forward<const std::valarray<double>>(_state.outputv);
      netinv = std::forward<const std::valarray<double>>(_state.netinv);
      rawinv = std::forward<const std::valarray<double>>(_state.rawinv);

      dy_dnet.set(std::forward<const Array2D<double>>(_state.dy_dnet));
      dnet_dx.set(std::forward<const Array2D<double>>(_state.dnet_dx));
      dnet_dw.set(std::forward<const Array2D<double>>(_state.dnet_dw));
      dE_dw.set(std::forward<const Array2D<double>>(_state.dE_dw));

      dE_dy = std::forward<const std::valarray<double>>(_state.dE_dy);
      dE_dnet = std::forward<const std::valarray<double>>(_state.dE_dnet);
      dE_dx = std::forward<const std::valarray<double>>(_state.dE_dx);
   }*/
}

#endif //FLEX_NEURALNET_LAYERSTATE_H_
