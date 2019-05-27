//
// Created by kfedrick on 5/16/19.
//

#ifndef FLEX_NEURALNET_PURELINTRANS_H_
#define FLEX_NEURALNET_PURELINTRANS_H_

#include <vector>
#include "Array.h"
#include "MyNetSum.h"
#include "Layer.h"

namespace flexnnet
{
   class PureLinTrans : public flexnnet::MyNetSum
   {

   public:
      PureLinTrans(unsigned int _sz);
      ~PureLinTrans();

      void set_gain (double _val);
      double get_gain (void) const;

   public:
      void calc_layer_output (std::vector<double> &_out, const std::vector<double> &_netin);
      const Array<double>& calc_dAdN (const std::vector<double> &_out, const std::vector<double> &_netin);

   protected:
      void resize(unsigned int _layer_sz, unsigned int _rawin_sz);

   private:
      unsigned int tranfersvec_size;
      double gain;

   private:
      Array<double> dAdN;
   };

   inline void PureLinTrans::set_gain (double _val)
   {
      gain = _val;
   }

   inline double PureLinTrans::get_gain (void) const
   {
      return gain;
   }

   inline void PureLinTrans::resize(unsigned int _layer_sz, unsigned int _rawin_sz)
   {
      MyNetSum::resize(_layer_sz, _rawin_sz);
   }

}

#endif //FLEX_NEURALNET_PURELINTRANS_H_
