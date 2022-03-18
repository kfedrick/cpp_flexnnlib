//
// Created by kfedrick on 3/16/22.
//

#ifndef FLEX_NEURALNET_TDCOSTTOGOLOSSFUNC_H_
#define FLEX_NEURALNET_TDCOSTTOGOLOSSFUNC_H_

#include <flexnnet.h>
#include <Exemplar.h>

namespace flexnnet
{
   template<class InTyp,
      class TgtTyp, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class DataSet> class TDCostToGoLossFunc
   {
      using NNTyp = NN<InTyp, TgtTyp>;
      using DatasetTyp = DataSet<InTyp, TgtTyp, ExemplarSeries>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;

   public:
      static constexpr double DEFAULT_SUBSAMPLE_SZ = 0;

   public:
      double calc_fitness(
         NNTyp& _nnet, const DatasetTyp& _tstset,
         unsigned int _subsample_sz = DEFAULT_SUBSAMPLE_SZ);

      double evaluate_sample(
         NNTyp& _nnet, const ExemplarSeries<InTyp, TgtTyp>& _series);

      double calc_tde_gradient(
         const TgtTyp& _tgt_t1, const TgtTyp& _est_t1, const TgtTyp& _est_t0, ValarrMap& _err, double _E = 1);

   private:
      double max_r;
      std::vector<double> tde_vec;

   };

   template<class InTyp,
      class TgtTyp, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class DataSet> double TDCostToGoLossFunc<InTyp, TgtTyp, NN, DataSet>::calc_fitness(
      NNTyp& _nnet, const DatasetTyp& _tstset, unsigned int _subsample_sz)
   {
      _tstset.randomize_order();

      double series_sumsqr_tde;

      //double dataset_sumsqr_tde = 0;

      max_r = 0;
      tde_vec.clear();

      unsigned int effective_subsample_sz = (_subsample_sz>0) ? _subsample_sz : _tstset.size();
      if (effective_subsample_sz > _tstset.size())
         effective_subsample_sz = _tstset.size();

      size_t sample_no = 0, step_count = 0;
      for (const ExemplarSeries<InTyp, TgtTyp>& sample_series: _tstset)
      {
         if (sample_no >= effective_subsample_sz)
            break;

         evaluate_sample(_nnet, sample_series);

         //dataset_sumsqr_tde += series_sumsqr_tde;
         step_count += sample_series.size();
      }

      double dataset_sumsqr_tde = 0;
      for (double tde : tde_vec)
      {
         dataset_sumsqr_tde += (tde/max_r * tde/max_r);
      }

      return (tde_vec.size() > 0) ? (dataset_sumsqr_tde / tde_vec.size()) : 0;

      //return (step_count > 0) ? (sqrt(dataset_sumsqr_tde) / step_count) : 0;;
   }

   template<class InTyp,
      class TgtTyp, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class DataSet> double TDCostToGoLossFunc<InTyp, TgtTyp, NN, DataSet>::evaluate_sample(
      NNTyp& _nnet, const ExemplarSeries<InTyp, TgtTyp>& _series)
   {
      //std::cout << "TDEvaluator.evaluate_series()\n";

      ValarrMap tde_gradient = _series[0].second.value_map();
      const ExemplarTyp& exemplar = _series[0];

      TgtTyp nnout0 = _nnet.activate(exemplar.first);

      double sum_tde;

      /*
       * Sum of the squared temporal difference error across all steps
       * of all series in the sub-sampling.
       */
      double series_sumsqr_tde = 0;

      int series_len = _series.size();
      for (int ndx = 1; ndx < series_len-1; ndx++)
      {
         const TgtTyp nnout0 = _nnet.value();

         const ExemplarTyp& exemplar = _series[ndx];
         const TgtTyp& nnout1 = _nnet.activate(exemplar.first);

         sum_tde = calc_tde_gradient(exemplar.second, nnout1, nnout0, tde_gradient);

         tde_vec.push_back(sum_tde);

         series_sumsqr_tde += sum_tde * sum_tde;

         //nnout0 = nnout1;
      }
      const ExemplarTyp& last_exemplar = _series[series_len-1];

      // Assign last examplar to zeronnout just to get the feature names
      TgtTyp zeronnout = last_exemplar.second;
      ValarrMap vm({{"output",{0}}});
      std::valarray<double> va = {0};
      zeronnout[0].decode(va);

      nnout0 = _nnet.value();

      sum_tde = calc_tde_gradient(exemplar.second, zeronnout, nnout0, tde_gradient);
      tde_vec.push_back(sum_tde);

      return series_sumsqr_tde;
   }

   template<class InTyp,
      class TgtTyp, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class DataSet>
   double TDCostToGoLossFunc<InTyp, TgtTyp, NN, DataSet>::calc_tde_gradient(
      const TgtTyp& _tgt_t1, const TgtTyp& _est_t1, const TgtTyp& _est_t0, ValarrMap& _egradient, double _E)
   {
      std::valarray<double> sqrdiff;

      const ValarrMap& tgt_vamap = _tgt_t1.value_map();
      const ValarrMap& est_t1_vamap = _est_t1.value_map();
      const ValarrMap& est_t0_vamap = _est_t0.value_map();

      // Sum of the temporal difference error across all reinforcement signals
      double sum_tde = 0;

      for (const auto& tgt: tgt_vamap)
      {
         const std::string id = tgt.first;
         _egradient[id] = tgt.second + 0.98 * est_t1_vamap.at(id) - est_t0_vamap.at(id);

         if (est_t0_vamap.at(id).max() > max_r)
            max_r = est_t0_vamap.at(id).max();

         sum_tde += _egradient[id].sum();
      }

      return sum_tde;
   }
}

#endif // FLEX_NEURALNET_TDCOSTTOGOLOSSFUNC_H_
