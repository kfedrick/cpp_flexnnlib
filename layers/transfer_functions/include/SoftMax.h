//
// Created by kfedrick on 5/27/19.
//

#ifndef FLEX_NEURALNET_SOFTMAX_H_
#define FLEX_NEURALNET_SOFTMAX_H_

#include "Array2D.h"
#include "NetSumLayer.h"

namespace flexnnet
{
   class SoftMax : public NetSumLayer
   {
   public:
      struct Parameters
      {
         double gain;
         bool rescaled_flag;
      };

   public:
      static const Parameters DEFAULT_PARAMS;

   public:
      SoftMax(size_t _sz, const std::string& _name, const Parameters& _params = DEFAULT_PARAMS);
      SoftMax(const SoftMax& _softmax);
      ~SoftMax();

      SoftMax& operator=(const SoftMax& _purelin);
      std::shared_ptr<BasicLayer> clone(void) const override;

      void set_gain(double _val);
      double get_gain(void) const;

      void set_rescaled(bool _val);
      bool is_rescaled(void) const;

      void set_params(const Parameters& _params);

      std::string toJson(void) const;

   protected:
      void calc_layer_output(const std::valarray<double>& _netin, std::valarray<double>& _layerval) override;
      void calc_dy_dnet(const std::valarray<double>& _outv, Array2D<double>& _dydnet) override;
      const SoftMax::Parameters& get_params(void) const;

   private:
      void copy(const SoftMax& _purelin);

   private:
      double lower_bound;
      double output_range;

      Parameters params;

   private:
      // Temporary valarray to hold exp of net input vector
      std::valarray<double> exp_netin;

      Array2D<double> dAdN;
   };

   inline void SoftMax::set_params(const Parameters& _val)
   {
      set_gain(_val.gain);
      set_rescaled(_val.rescaled_flag);
   }

   inline
   const SoftMax::Parameters& SoftMax::get_params(void) const
   {
      return params;
   }

   inline void SoftMax::set_gain(double _val)
   {
      params.gain = _val;
   }

   inline double SoftMax::get_gain(void) const
   {
      return params.gain;
   }

   inline void SoftMax::set_rescaled(bool _val)
   {
      params.rescaled_flag = _val;
      if (params.rescaled_flag)
      {
         lower_bound = -1.0;
         output_range = 2.0;
      }
      else
      {
         lower_bound = 0.0;
         output_range = 1.0;
      }
   }

   inline bool SoftMax::is_rescaled(void) const
   {
      return params.rescaled_flag;
   }

   inline std::string SoftMax::toJson(void) const
   {
      return "";
   }
}

#endif //FLEX_NEURALNET_SOFTMAX_H_
