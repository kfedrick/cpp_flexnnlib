//
// Created by kfedrick on 5/25/19.
//

#ifndef FLEX_NEURALNET_TANSIGTRANS_H_
#define FLEX_NEURALNET_TANSIGTRANS_H_

#include <vector>
#include "Array.h"
#include "MyNetSum.h"

namespace flexnnet
{
   class TanSigTrans  : public flexnnet::MyNetSum
   {
   public:
      TanSigTrans(unsigned int _sz);
      ~TanSigTrans();

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


   inline void TanSigTrans::set_gain (double _val)
   {
      gain = _val;
   }

   inline double TanSigTrans::get_gain (void) const
   {
      return gain;
   }

   inline void TanSigTrans::resize(unsigned int _layer_sz, unsigned int _rawin_sz)
   {
      MyNetSum::resize(_layer_sz, _rawin_sz);
   }

}

#endif //FLEX_NEURALNET_TANSIGTRANS_H_
