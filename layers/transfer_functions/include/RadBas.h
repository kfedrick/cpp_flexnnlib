//
// Created by kfedrick on 5/24/19.
//

#ifndef FLEX_NEURALNET_RADBAS_H_
#define FLEX_NEURALNET_RADBAS_H_

#include "Array2D.h"
#include "EuclideanDistLayer.h"
#include "LayerSerializer.h"

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
      RadBas(size_t _sz, const std::string& _name, NetworkLayerType _type, const Parameters& _params = DEFAULT_PARAMS);
      ~RadBas();

      void set_rescaled(bool _val);
      bool is_rescaled(void) const;
      void set_params(const Parameters& _params);

      std::string toJson(void) const;

   protected:
      const std::valarray<double>& calc_layer_output(const std::valarray<double>& _netin);
      const Array2D<double>& calc_dAdN(const std::valarray<double>& _out);

   private:
      double lower_bound;
      double output_range;

      Parameters params;
   };

   inline void RadBas::set_params(const Parameters& _val)
   {
      set_rescaled(_val.rescaled_flag);
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
      return LayerSerializer<RadBas>::toJson(*this);
   }
}

#endif //FLEX_NEURALNET_RADBAS_H_
