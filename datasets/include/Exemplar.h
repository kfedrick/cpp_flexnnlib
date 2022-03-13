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
      Exemplar() : std::pair<InTyp,TgtTyp>(), valid_extern_target(false)
      {
      }

      Exemplar(const InTyp& _in, const TgtTyp& _tgt, bool _externflg=true) : std::pair<InTyp,TgtTyp>(_in,_tgt)
      {
         valid_extern_target = _externflg;
      }

      bool valid_target() const
      {
         return valid_extern_target;
      }

   private:

      // Exemplar has a valid external training signal
      bool valid_extern_target;
   };
}

#endif //FLEX_NEURALNET_EXEMPLAR_H_
