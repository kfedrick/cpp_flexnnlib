//
// Created by kfedrick on 5/25/19.
//

#ifndef FLEX_NEURALNET_TANSIG_H_
#define FLEX_NEURALNET_TANSIG_H_

#include "Array2D.h"
#include "NetSumLayer.h"
#include "LayerSerializer.h"

namespace flexnnet
{
   class TanSig  : public NetSumLayer
   {
   public:
      struct Parameters
      {
         double gain;
      };

   public:
      static const Parameters DEFAULT_PARAMS;

   public:
      TanSig(size_t _sz, const std::string &_name, NetworkLayerType _type, const Parameters& _params = DEFAULT_PARAMS);
      ~TanSig();

      void set_gain(double _val);
      double get_gain(void) const;
      void set_params(const Parameters& _params);

      std::string toJson (void) const;

   protected:
      const std::valarray<double>& calc_layer_output (const std::valarray<double> &_netin);
      const Array2D<double>& calc_dAdN (const std::valarray<double> &_out);

   private:
      Parameters params;
   };

   inline void TanSig::set_params (const Parameters& _val)
   {
      params = _val;
   }

   inline void TanSig::set_gain (double _val)
   {
      params.gain = _val;
   }

   inline double TanSig::get_gain (void) const
   {
      return params.gain;
   }

   inline std::string TanSig::toJson (void) const
   {
      return LayerSerializer<TanSig>::toJson (*this);
   }
}

#endif //FLEX_NEURALNET_TANSIG_H_