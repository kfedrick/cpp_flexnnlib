//
// Created by kfedrick on 5/19/19.
//

#ifndef FLEX_NEURALNET_LOGSIG_H_
#define FLEX_NEURALNET_LOGSIG_H_

#include <memory>
#include "Array2D.h"
#include "NetSumLayer.h"


namespace flexnnet
{
   class LogSig : public NetSumLayer
   {

   public:
      struct Parameters
      {
         double gain;
      };

   public:
      static const Parameters DEFAULT_PARAMS;

   public:
      LogSig(size_t _sz, const std::string& _name, const Parameters& _params = DEFAULT_PARAMS);
      LogSig(const LogSig& _logsig);
      ~LogSig();

      LogSig& operator=(const LogSig& _purelin);
      std::shared_ptr<BasicLayer> clone(void) const override;

      void set_gain(double _val);
      double get_gain(void) const;
      void set_params(const Parameters& _params);

      std::string toJson(void) const;

   protected:
      void calc_layer_output(const std::valarray<double>& _netin, std::valarray<double>& _layerval) override;
      void calc_dy_dnet(const std::valarray<double>& _outv, Array2D<double>& _dydnet) override;
      const LogSig::Parameters& get_params(void) const;

   private:
      void copy(const LogSig& _logsig);

   private:
      Parameters params;
   };

   inline void LogSig::set_params(const Parameters& _val)
   {
      params = _val;
   }

   inline
   const LogSig::Parameters& LogSig::get_params(void) const
   {
      return params;
   }

   inline void LogSig::set_gain(double _val)
   {
      params.gain = _val;
   }

   inline double LogSig::get_gain(void) const
   {
      return params.gain;
   }

   inline std::string LogSig::toJson(void) const
   {
      return "";
   }
}

#endif //FLEX_NEURALNET_LOGSIG_H_
