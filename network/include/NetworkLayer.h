//
// Created by kfedrick on 4/9/21.
//

#ifndef FLEX_NEURALNET_NETWORKLAYER_H_
#define FLEX_NEURALNET_NETWORKLAYER_H_

#include <memory>
#include <string>
#include <valarray>

#include <flexnnet.h>
#include <LayerConnRecord.h>
#include <BaseNetworkLayer.h>
#include <ExternalInputRecord.h>

namespace flexnnet
{
   class NeuralNetTopology;
   class NetworkLayer;

   class NetworkLayer : public BaseNetworkLayer
   {
      friend class NeuralNetTopology;
      friend class NeuralNetBuilder;

   protected:
      NetworkLayer(bool _is_output = false);
      NetworkLayer(const NetworkLayer& _layer);

   public:
      ~NetworkLayer();
      virtual std::shared_ptr<NetworkLayer> clone(void) = 0;

      /**
       * Return layer id.
       * @return
       */
      virtual const std::string& name() const = 0;

      /**
       * Returns the size of the layer activity vector.
       * @return
       */
      virtual size_t
      size() const = 0;

      /**
       * Return the size of the input vector for this layer
       * @return
       */
      virtual size_t input_size() const = 0;

      // Return true if this is an output basic_layer
      virtual bool
      is_output_layer(void) const;

      // Return the activation vectorize vector of this layer.
      virtual const std::valarray<double>&
      value() const;

      virtual const LayerWeights& weights() const = 0;

      virtual void initialize_weights(void) = 0;

      virtual void set_weights(double _val) = 0;

      virtual void set_weights(const Array2D<double>& _weights) = 0;

      virtual void adjust_weights(const Array2D<double>& _weights) = 0;

      virtual const Array2D<double>& dEdw(void) const;

      void set_weight_initializer(std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)>& _func);

      const std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)>& get_weight_initializer(void) const;


      /**
       * Marshal layer inputs, activate the base layer and return the
       * layer output.
       * @param _externin
       * @return
       */
      virtual const std::valarray<double>&
      activate(const ValarrMap& _externin) = 0;

      virtual const std::valarray<double>&
      backprop(const ValarrMap& _externerror) = 0;

   protected:

      /**
       * Marshal layer and external inputs into a single valarray.
       * @param _externin
       * @return
       */
      void
      concat_inputs(const ValarrMap& _externin, std::valarray<double>& _invec);

      const ValarrMap&
      marshal_inputs(const ValarrMap& _externin);

      /**
       * Marshal the external and back-propagated errors to calculate
       * the cumulative external error vector for this layer.
       *
       * @param _externerr
       * @return
       */
      void
      gather_error(const ValarrMap& _externerr, std::valarray<double>& _dEdy);

      void
      scatter_input_error(const std::valarray<double>& _dEdx);

      size_t
      append_virtual_vector(const std::valarray<double>& _srcvec, size_t& _vindex, std::valarray<double>& _tgtvec);

   protected:
      LayerState layer_state;

   private:
      bool output_layer_flag;

      ValarrMap input_map;

      std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)> weight_initializer_func;
   };

   inline
   bool
   NetworkLayer::is_output_layer(void) const
   {
      return output_layer_flag;
   }

   inline
   const std::valarray<double>& NetworkLayer::value() const
   {
      return layer_state.outputv;
   }

   inline
   void NetworkLayer::set_weight_initializer(std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)>& _func)
   {
      weight_initializer_func = _func;
   }

   inline
   const std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)>& NetworkLayer::get_weight_initializer(void) const
   {
      return weight_initializer_func;
   }

   inline
   const Array2D<double>& NetworkLayer::dEdw(void) const
   {
      return layer_state.dE_dw;
   }
}

#endif //FLEX_NEURALNET_NETWORKLAYER_H_
