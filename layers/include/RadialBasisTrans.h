//
// Created by kfedrick on 5/24/19.
//

#ifndef FLEX_NEURALNET_RADIALBASISTRANS_H_
#define FLEX_NEURALNET_RADIALBASISTRANS_H_

#include <vector>
#include "Array.h"
#include "Layer.h"
#include "MyEuclideanDist.h"

namespace flexnnet
{
   class RadialBasisTrans  : public flexnnet::MyEuclideanDist
   {
   public:
      RadialBasisTrans(unsigned int _sz);
      ~RadialBasisTrans();

   public:
      void calc_layer_output (std::vector<double>& _out, const std::vector<double>& _netin);
      const Array<double>& calc_dAdN(const std::vector<double>& _out, const std::vector<double>& _netin);

   protected:
      void resize(unsigned int _layer_sz, unsigned int _rawin_sz);

   private:
      Array<double> dAdN;
   };

   inline void RadialBasisTrans::resize(unsigned int _layer_sz, unsigned int _rawin_sz)
   {
      MyEuclideanDist::resize(_layer_sz, _rawin_sz);
   }
}

#endif //FLEX_NEURALNET_RADIALBASISTRANS_H_
