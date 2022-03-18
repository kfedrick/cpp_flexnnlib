//
// Created by kfedrick on 3/13/22.
//

#ifndef FLEX_NEURALNET_RMSELOSSFUNC_H_
#define FLEX_NEURALNET_RMSELOSSFUNC_H_

#include <LossFunction.h>
#include <Exemplar.h>

namespace flexnnet
{
   template<class InTyp,
      class TgtTyp, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class DataSet> class RMSELossFunc : public LossFunction<InTyp, TgtTyp, Exemplar, NN, DataSet>
   {
      using NNTyp = NN<InTyp, TgtTyp>;
      using DatasetTyp = DataSet<InTyp, TgtTyp, Exemplar>;

   public:
      double calc_fitness(NNTyp& _nnet, const DatasetTyp& _tstset, unsigned int _subsample_sz=LossFunction<InTyp, TgtTyp, Exemplar, NN, DataSet>::DEFAULT_FITNESS_SUBSAMPLE_SZ);
      double
      evaluate_sample(NNTyp& _nnet, const Exemplar<InTyp,TgtTyp>& _sample, TgtTyp& _est, ValarrMap& _egradient);
      virtual double calc_dEde(const TgtTyp& _tgt, const TgtTyp& _est, ValarrMap& _err, double _E=1);
   };

   template<class InTyp,
      class TgtTyp, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class DataSet>
   double RMSELossFunc<InTyp,
                       TgtTyp,
                       NN,
                       DataSet>::calc_fitness(NNTyp& _nnet, const DatasetTyp& _tstset, unsigned int _subsample_sz)
   {
      double rmse, sse = 0;

      unsigned int effective_subsample_sz = (_subsample_sz>0) ? _subsample_sz : _tstset.size();
      if (effective_subsample_sz > _tstset.size())
         effective_subsample_sz = _tstset.size();

      size_t sample_count = 0;
      for (const Exemplar<InTyp,TgtTyp>& it : _tstset)
      {
         TgtTyp nnout = it.second;
         ValarrMap egradient = nnout.value_map();

         if (sample_count >= effective_subsample_sz)
            break;

         sse += evaluate_sample(_nnet, it, nnout, egradient);
         sample_count++;
      }

      rmse = (sample_count > 0) ? (0.5*sse / sample_count) : 0;
      return (rmse > 0) ? sqrt(rmse) : 0;
   }

   template<class InTyp,
      class TgtTyp, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class DataSet>
   double
   RMSELossFunc<InTyp,
                TgtTyp,
                NN,
                DataSet>::evaluate_sample(NNTyp& _nnet, const Exemplar<InTyp,TgtTyp>& _sample, TgtTyp& _est, ValarrMap& _egradient)

{
      _est = _nnet.activate(_sample.first);
      return calc_dEde(_sample.second, _est, _egradient);
   }

   template<class InTyp,
      class TgtTyp, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class DataSet>
   double RMSELossFunc<InTyp,
                       TgtTyp,
                       NN,
                       DataSet>::calc_dEde(const TgtTyp& _tgt, const TgtTyp& _est, ValarrMap& _err, double _E)
   {
      ValarrMap temp = _tgt.value_map();

      const ValarrMap& tgt_va = _tgt.value_map();
      const ValarrMap& est_va = _est.value_map();

      double sse = 0;
      for (const auto& it : tgt_va)
      {
         _err[it.first] = -(it.second - est_va.at(it.first));
         temp[it.first] = _err[it.first] * _err[it.first];

         sse += temp[it.first].sum();
      }
      return sse;
   }
}

#endif // FLEX_NEURALNET_RMSELOSSFUNC_H_
