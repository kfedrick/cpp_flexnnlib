//
// Created by kfedrick on 5/16/19.
//

#ifndef FLEX_NEURALNET_PURELIN_H_
#define FLEX_NEURALNET_PURELIN_H_

#include <vector>
#include "Array2D.h"
#include "NetSumLayer.h"

namespace flexnnet
{
   class PureLin : public NetSumLayer
   {
   public:
      struct Parameters
      {
         double gain;
      };

   public:
      static const Parameters DEFAULT_PARAMS;

   public:
      PureLin(size_t _sz, const std::string& _id, const Parameters& _params = DEFAULT_PARAMS);
      PureLin(const PureLin& _purelin);
      ~PureLin();

      PureLin& operator=(const PureLin& _purelin);
      std::shared_ptr<BasicLayer> clone(void) const override;

      void set_gain(double _val);
      double get_gain(void) const;
      void set_params(const Parameters& _params);

      std::string toJson(void) const;

   protected:
      const std::valarray<double>& calc_layer_output(const std::valarray<double>& _netin);
      const Array2D<double>& calc_dy_dnet(const std::valarray<double>& _out);

   private:
      void copy(const PureLin& _purelin);

   private:
      Parameters params;
   };

   inline void PureLin::set_params(const Parameters& _val)
   {
      params = _val;
   }

   inline void PureLin::set_gain(double _val)
   {
      params.gain = _val;
   }

   inline double PureLin::get_gain(void) const
   {
      return params.gain;
   }

   inline std::string PureLin::toJson(void) const
   {
      return "";
   }
}

#endif //FLEX_NEURALNET_PURELIN_H_
