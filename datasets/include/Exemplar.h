//
// Created by kfedrick on 4/18/21.
//

#ifndef FLEX_NEURALNET_EXEMPLAR_H_
#define FLEX_NEURALNET_EXEMPLAR_H_

#include <utility>

namespace flexnnet
{
   template<class InTyp, class TgtTyp>
   class Exemplar : public std::pair<InTyp, TgtTyp>
   {
   public:
      Exemplar() : std::pair<InTyp,TgtTyp>()
      {
      }

      Exemplar(const InTyp& _in, const TgtTyp& _tgt) : std::pair<InTyp,TgtTyp>(_in,_tgt)
      {
      }
   };
}

#endif //FLEX_NEURALNET_EXEMPLAR_H_
