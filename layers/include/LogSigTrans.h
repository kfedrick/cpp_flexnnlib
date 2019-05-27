//
// Created by kfedrick on 5/19/19.
//

#ifndef FLEX_NEURALNET_LOGSIGTRANS_H_
#define FLEX_NEURALNET_LOGSIGTRANS_H_

#include <vector>
#include "Array.h"
#include "MyNetSum.h"
#include "Layer.h"

namespace flexnnet
{
   class LogSigTrans  : public flexnnet::MyNetSum
   {
   public:
      LogSigTrans(unsigned int _sz);
      ~LogSigTrans();

      void set_gain(double _val);
      double get_gain(void) const;

   public:
      void calc_layer_output (std::vector<double>& _out, const std::vector<double>& _netin);
      const Array<double>& calc_dAdN(const std::vector<double>& _out, const std::vector<double>& _netin);

   protected:
      void resize(unsigned int _layer_sz, unsigned int _rawin_sz);

   private:
      double gain;

   private:
      Array<double> dAdN;
   };


   inline void LogSigTrans::set_gain(double _val)
   {
      gain = _val;
   }

   inline double LogSigTrans::get_gain(void) const
   {
      return gain;
   }

   inline void LogSigTrans::resize(unsigned int _layer_sz, unsigned int _rawin_sz)
   {
      MyNetSum::resize(_layer_sz, _layer_sz);
   }
}

#endif //FLEX_NEURALNET_LOGSIGTRANS_H_
