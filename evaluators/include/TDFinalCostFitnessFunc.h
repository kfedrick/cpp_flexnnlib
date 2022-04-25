//
// Created by kfedrick on 3/15/22.
//

#ifndef FLEX_NEURALNET_TDFINALCOSTFITNESSFUNC_H_
#define FLEX_NEURALNET_TDFINALCOSTFITNESSFUNC_H_

#include <flexnnet.h>
#include <Exemplar.h>
#include <FitnessFunc.h>

namespace flexnnet
{
   template<class InTyp, class TgtTyp, template<class, class> class Sample=ExemplarSeries>
   class TDFinalCostFitnessFunc : public FitnessFunc<InTyp, TgtTyp, Sample>
   {
      using NNTyp = NeuralNet<InTyp, TgtTyp>;
      using DatasetTyp = DataSet<InTyp, TgtTyp, ExemplarSeries>;
      using SampleTyp = Sample<InTyp, TgtTyp>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;
      using ExemplarSeriesTyp = ExemplarSeries<InTyp, TgtTyp>;

   public:
      double calc_fitness(
         NNTyp& _nnet, const DatasetTyp& _nnout, unsigned int _subsample_sz = FitnessFunc<InTyp,
                                                                                           TgtTyp,ExemplarSeries>::DEFAULT_FITNESS_SUBSAMPLE_SZ);
      double evaluate_sample(
         NNTyp& _nnet, const ExemplarSeriesTyp& _series);

      double calc_tde_gradient(
         const TgtTyp& _tgt_t1, const TgtTyp& _est_t0, ValarrMap& _err, double _E = 1);

   private:
      std::vector<double> tde_cache;
      double min_r, max_r, range_r;
   };

   template<class InTyp, class TgtTyp, template<class, class> class Sample>
   double TDFinalCostFitnessFunc<InTyp, TgtTyp, Sample>::calc_fitness(
      NNTyp& _nnet, const DatasetTyp& _tstset, unsigned int _subsample_sz)
   {
      _tstset.randomize_order();

      double series_sumsqr_tde;

      double dataset_sumsqr_tde = 0;

      tde_cache.clear();
      min_r = std::numeric_limits<double>::max();
      max_r = 0;

      unsigned int effective_subsample_sz = (_subsample_sz > 0) ? _subsample_sz : _tstset.size();
      if (effective_subsample_sz > _tstset.size())
         effective_subsample_sz = _tstset.size();

      size_t sample_no = 0, step_count = 0;
      for (const SampleTyp& sample_series: _tstset)
      {
         if (sample_no >= effective_subsample_sz)
            break;

         evaluate_sample(_nnet, sample_series);
      }

      range_r = max_r - min_r;
      dataset_sumsqr_tde = 0;
      for (double tde : tde_cache)
         dataset_sumsqr_tde += ((tde/range_r) * (tde/range_r));
      return (tde_cache.size() > 0) ? (sqrt(dataset_sumsqr_tde) / tde_cache.size()) : 0;
   }

   template<class InTyp, class TgtTyp, template<class, class> class Sample>
   double TDFinalCostFitnessFunc<InTyp, TgtTyp, Sample>::evaluate_sample(
      NNTyp& _nnet, const ExemplarSeriesTyp& _series)
   {
      //std::cout << "TDEvaluator.evaluate_series()\n";

      ValarrMap tde_gradient = _series[0].second.value_map();
      const ExemplarTyp& exemplar = _series[0];

      _nnet.activate(exemplar.first);

      double sum_tde;

      /*
       * Sum of the squared temporal difference error across all steps
       * of all series in the sub-sampling.
       */
      double series_sumsqr_tde = 0;

      int series_len = _series.size();
      for (int ndx = 1; ndx < series_len; ndx++)
      {
         const TgtTyp nnout0 = _nnet.value();

         const ExemplarTyp& exemplar = _series[ndx];
         const TgtTyp nnout1 = _nnet.activate(exemplar.first);

         if (exemplar.valid_target())
            sum_tde = calc_tde_gradient(exemplar.second, nnout0, tde_gradient);
         else
            sum_tde = calc_tde_gradient(nnout1, nnout0, tde_gradient);

         tde_cache.push_back(sum_tde);
         series_sumsqr_tde += sum_tde * sum_tde;
      }

      return series_sumsqr_tde;
   }

   template<class InTyp, class TgtTyp, template<class, class> class Sample>
   double TDFinalCostFitnessFunc<InTyp, TgtTyp, Sample>::calc_tde_gradient(
      const TgtTyp& _tgt_t1, const TgtTyp& _est_t0, ValarrMap& _egradient, double _E)
   {
      const ValarrMap& tgt_vamap = _tgt_t1.value_map();
      const ValarrMap& prev_est_vamap = _est_t0.value_map();

      // Sum of the temporal difference error across all reinforcement signals
      double sum_tde = 0;

      for (const auto& tgt: tgt_vamap)
      {
         const std::string id = tgt.first;
         _egradient[id] = tgt.second - prev_est_vamap.at(id);

         if (prev_est_vamap.at(id).max() > max_r)
            max_r = prev_est_vamap.at(id).max();

         if (prev_est_vamap.at(id).min() < min_r)
            min_r = prev_est_vamap.at(id).min();

         sum_tde += _egradient[id].sum();
      }

      return sum_tde;
   }
}

#endif // FLEX_NEURALNET_TDFINALCOSTFITNESSFUNC_H_
