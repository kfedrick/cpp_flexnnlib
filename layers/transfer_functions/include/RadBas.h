//
// Created by kfedrick on 5/24/19.
//

#ifndef FLEX_NEURALNET_RADBAS_H_
#define FLEX_NEURALNET_RADBAS_H_

#include "Array2D.h"
#include "EuclideanDistLayer.h"

namespace flexnnet
{
   class RadBas : public EuclideanDistLayer
   {
   public:
      struct Parameters
      {
         bool rescaled_flag;
      };

   public:
      static const Parameters DEFAULT_PARAMS;

   public:
      RadBas(size_t _sz, const std::string& _name, const Parameters& _params = DEFAULT_PARAMS);
      RadBas(const RadBas& _radbas);
      ~RadBas();

      RadBas& operator=(const RadBas& _radbas);
      std::shared_ptr<BasicLayer> clone(void) const override;

      void set_rescaled(bool _val);
      bool is_rescaled(void) const;
      void set_params(const Parameters& _params);

      std::string toJson(void) const;

   protected:
      void calc_layer_output(const std::valarray<double>& _netin, std::valarray<double>& _layerval) override;
      void calc_dy_dnet(const std::valarray<double>& _outv, Array2D<double>& _dydnet) override;
      const RadBas::Parameters& get_params(void) const;


   private:
      void copy(const RadBas& _radbas);

   private:
      double lower_bound;
      double output_range;

      Parameters params;
   };

   inline void RadBas::set_params(const Parameters& _val)
   {
      set_rescaled(_val.rescaled_flag);
   }

   inline
   const RadBas::Parameters& RadBas::get_params(void) const
   {
      return params;
   }

   inline void RadBas::set_rescaled(bool _val)
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

   inline bool RadBas::is_rescaled(void) const
   {
      return params.rescaled_flag;
   }

   inline std::string RadBas::toJson(void) const
   {
      return "";
   }
}

#endif //FLEX_NEURALNET_RADBAS_H_
