//
// Created by kfedrick on 3/17/22.
//

#ifndef FLEX_NEURALNET_FFBACKPROPALGO_H_
#define FLEX_NEURALNET_FFBACKPROPALGO_H_

#include <BaseTrainingAlgo.h>

namespace flexnnet
{
   /**
 * Deep NetworkReinforcement Learning Algorithm
 *
 * @tparam InTyp - Neural network input data typename
 * @tparam TgtTyp - Neural net training set target data typename
 * @tparam NN - Neural network class name
 * @tparam Dataset - Dataset container class name
 * @tparam FitFunc - Fitness function class name
 * @tparam LRPolicy - Learning rate policy class name
 */
   template<class InTyp,
      class TgtTyp, template<class, class>
      class Sample, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class Dataset, template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class>
      class FitFunc,
      class LRPolicy> class FFBackpropAlgo : public BaseTrainingAlgo<InTyp,
                                                                     TgtTyp,
                                                                     Sample,
                                                                     NN,
                                                                     Dataset,
                                                                     FitFunc,
                                                                     LRPolicy>
   {
      using DatasetTyp = Dataset<InTyp, TgtTyp, ExemplarSeries>;
      using ExemplarSeriesTyp = ExemplarSeries<InTyp, TgtTyp>;
      using SampleTyp = Sample<InTyp, TgtTyp>;
      using ExemplarTyp = Exemplar<InTyp, TgtTyp>;

   public:
      FFBackpropAlgo(NN<InTyp, TgtTyp>& _nnet);

   protected:
      void train_sample(const SampleTyp& _sample);

      /**
       * Calculate the neural network weight updates given the specified
       * output error gradient.
       *
       * Precondition:
       *    The network layers retain the state information
       *    for the input/target training exemplar used to generate
       *    the error gradient.
       *
       * @param _egradient
       */
      void calc_weight_updates(const NN<InTyp, TgtTyp>& _nnet, const ValarrMap& _egradient);

   protected:
      std::map<std::string, Array2D<double>> weight_updates;

   };

   template<class InTyp,
      class TgtTyp, template<class, class>
      class Sample, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class Dataset, template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class>
      class FitFunc,
      class LRPolicy>
   FFBackpropAlgo<InTyp, TgtTyp, Sample, NN, Dataset, FitFunc, LRPolicy>::FFBackpropAlgo(
      NN<InTyp, TgtTyp>& _nnet)
      : BaseTrainingAlgo<InTyp, TgtTyp, Sample, NN, Dataset, FitFunc, LRPolicy>(_nnet)
   {

      weight_updates.clear();

      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = this->nnet.get_layers();
      for (auto it: layers)
      {
         std::string id = it.first;
         const LayerWeights& w = it.second->weights();

         Array2D<double>::Dimensions dim = w.const_weights_ref.size();

         weight_updates[id] = {};
         weight_updates[id].resize(dim.rows, dim.cols);
      }
   }

   template<class InTyp,
      class TgtTyp, template<class, class>
      class Sample, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class Dataset, template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class>
      class FitFunc,
      class LRPolicy>
   void FFBackpropAlgo<InTyp, TgtTyp, Sample, NN, Dataset, FitFunc, LRPolicy>::train_sample(
      const SampleTyp& _exemplar)
   {
      //std::cout << "FFBackpropAlgo.train_sample()\n" << std::flush;

      const NNFeatureSet<TgtTyp>& nn_out = this->nnet.activate(_exemplar.first);

      //const std::map<std::string, std::valarray<double>>& nnoutv_map = nn_out.value_map();
      const std::map<std::string, std::valarray<double>>
         & targetv_map = _exemplar.second.value_map();

      ValarrMap gradient;
      this->fitnessfunc.calc_dEde(_exemplar.second, nn_out, gradient);
      this->calc_weight_updates(this->nnet, gradient);
      LRPolicy::calc_learning_rate_adjustment(0);
   }

   template<class InTyp,
      class TgtTyp, template<class, class>
      class Sample, template<class, class>
      class NN, template<class, class, template<class, class> class>
      class Dataset, template<class, class, template<class, class> class,
      template<class, class, template<class, class> class> class>
      class FitFunc,
      class LRPolicy>
   void FFBackpropAlgo<InTyp, TgtTyp, Sample, NN, Dataset, FitFunc, LRPolicy>::calc_weight_updates(
      const NN<InTyp, TgtTyp>& _nnet, const ValarrMap& _egradient)
   {
      this->nnet.backprop(_egradient);

      const std::map<std::string, std::shared_ptr<NetworkLayer>>& layers = this->nnet.get_layers();
      for (auto it: layers)
      {
         std::string id = it.first;
         Array2D<double> lr = LRPolicy::get_learning_rates(id);

         const Array2D<double> dE_dw = it.second->dEdw();

         const Array2D<double>::Dimensions dims = this->weight_updates[id].size();

         // If this layer doesn't train biases, stop before the last column
         unsigned int last_col = (TrainerConfig::train_biases(id)) ? dims.cols : dims.cols - 1;

         this->weight_updates[id] = 0;
         for (unsigned int row = 0; row < dims.rows; row++)
            for (unsigned int col = 0; col < last_col; col++)
               this->weight_updates[id].at(row, col) = -lr.at(row, col) * dE_dw.at(row, col);

         this->accumulate_weight_updates(this->nnet, id, this->weight_updates[id]);
      }
   }
}

#endif // FLEX_NEURALNET_FFBACKPROPALGO_H_
