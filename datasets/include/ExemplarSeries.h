//
// Created by kfedrick on 4/18/21.
//

#ifndef FLEX_NEURALNET_EXEMPLARSERIES_H_
#define FLEX_NEURALNET_EXEMPLARSERIES_H_

#include <vector>
#include <Exemplar.h>

namespace flexnnet
{
   template<class InTyp, class TgtTyp>
   class ExemplarSeries : public std::vector<Exemplar<InTyp, TgtTyp>>
   {
   public:
      ExemplarSeries();
   };

   template<class InTyp, class TgtTyp>
   inline
   ExemplarSeries<InTyp, TgtTyp>::ExemplarSeries() : std::vector<Exemplar<InTyp, TgtTyp>>()
   {
   }

} // end namespace flexnnet

#endif //FLEX_NEURALNET_EXEMPLARSERIES_H_
