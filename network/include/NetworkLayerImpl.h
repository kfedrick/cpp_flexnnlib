//
// Created by kfedrick on 4/9/21.
//

#ifndef FLEX_NEURALNET_NETWORKLAYERIMPL_H_
#define FLEX_NEURALNET_NETWORKLAYERIMPL_H_

#include <string>

#include <flexnnet.h>
#include <NetworkLayer.h>
#include <BaseNetworkLayer.h>
#include <LayerState.h>

namespace flexnnet
{
   template<typename _TLAYER>
   class NetworkLayerImpl : public NetworkLayer, protected _TLAYER
   {

   public:
      NetworkLayerImpl(size_t _sz, const std::string& _name, const typename _TLAYER::Parameters& _params = _TLAYER::DEFAULT_PARAMS, bool _oflag = false);
      NetworkLayerImpl(const NetworkLayerImpl<_TLAYER>& _layer);
      ~NetworkLayerImpl();

      virtual std::shared_ptr<NetworkLayer> clone(void);

      /**
       * Return layer id.
       * @return
       */
      virtual const std::string& name() const;

      /**
       * Returns the size of the layer activity vector.
       * @return
       */
      virtual size_t size() const;

      virtual size_t input_size() const;

      virtual const LayerWeights& weights() const;

      virtual void initialize_weights(void);

      virtual void set_weights(double _val);

      virtual void set_weights(const Array2D<double>& _weights);

      virtual void adjust_weights(const Array2D<double>& _weights);

      /**
       * Marshal layer inputs, activate the base layer and return the
       * layer output.
       * @param _externin
       * @return
       */
      virtual const std::valarray<double>& activate(const ValarrMap& _externin);

      virtual const std::valarray<double>& backprop(const ValarrMap& _externerror);

   protected:
      virtual void set_input_size(size_t _rawin_sz);
   };

   template<typename _TLAYER>
   inline
   NetworkLayerImpl<_TLAYER>::NetworkLayerImpl(size_t _sz, const std::string& _name, const typename _TLAYER::Parameters& _params, bool _oflag) : NetworkLayer(_oflag), _TLAYER(_sz, _name, _params)
   {
      // -- No body
   }

   template<typename _TLAYER>
   inline
   NetworkLayerImpl<_TLAYER>::NetworkLayerImpl(const NetworkLayerImpl<_TLAYER>& _layer) : NetworkLayer(_layer), _TLAYER(_layer)
   {
   }

   template<typename _TLAYER>
   inline
   NetworkLayerImpl<_TLAYER>::~NetworkLayerImpl()
   {
      // -- No body
   }

   template<typename _TLAYER>
   inline
   std::shared_ptr<NetworkLayer> NetworkLayerImpl<_TLAYER>::clone(void)
   {
      auto ptr = std::make_shared<NetworkLayerImpl<_TLAYER>>(NetworkLayerImpl<_TLAYER>(*this));
      return ptr;
   }

   template<typename _TLAYER>
   inline
   const std::valarray<double>& NetworkLayerImpl<_TLAYER>::activate(const ValarrMap& _externin)
   {
      //std::cout << "NetworkLayerImpl.activate()\n" << std::flush;

      concat_inputs(_externin, layer_state.rawinv);
      _TLAYER::activate(layer_state.rawinv, layer_state);

      //std::cout << "NetworkLayerImpl.activate() EXIT\n" << std::flush;
      return layer_state.outputv;
   }

   template<typename _TLAYER>
   inline
   const std::valarray<double>& NetworkLayerImpl<_TLAYER>::backprop(const ValarrMap& _externerror)
   {
      gather_error(_externerror, layer_state.dE_dy);
      _TLAYER::backprop(layer_state.dE_dy, layer_state);
      scatter_input_error(layer_state.dE_dx);

      return layer_state.dE_dx;
   }

   template<typename _TLAYER>
   inline
   const std::string& NetworkLayerImpl<_TLAYER>::name() const
   {
      return _TLAYER::name();
   }

   template<typename _TLAYER>
   inline
   size_t NetworkLayerImpl<_TLAYER>::size() const
   {
      return _TLAYER::size();
   }

   template<typename _TLAYER>
   inline
   size_t NetworkLayerImpl<_TLAYER>::input_size() const
   {
      return _TLAYER::input_size();
   }

   template<typename _TLAYER>
   inline
   const LayerWeights& NetworkLayerImpl<_TLAYER>::weights() const
   {
      return _TLAYER::weights();
   }

   template<typename _TLAYER>
   inline
   void NetworkLayerImpl<_TLAYER>::set_weights(double _val)
   {
      _TLAYER::weights().set(_val);
   }

   template<typename _TLAYER>
   inline
   void NetworkLayerImpl<_TLAYER>::set_weights(const Array2D<double>& _weights)
   {
      _TLAYER::weights().set(_weights);
   }

   template<typename _TLAYER>
   inline
   void NetworkLayerImpl<_TLAYER>::adjust_weights(const Array2D<double>& _weights)
   {
      _TLAYER::weights().adjust_weights(_weights);
   }

   template<typename _TLAYER>
   inline
   void NetworkLayerImpl<_TLAYER>::initialize_weights(void)
   {
      const std::function<Array2D<double>(unsigned int _rows, unsigned int _cols)>& ifunc = get_weight_initializer();
      if (ifunc)
      {
         Array2D<double>::Dimensions dims = weights().size();
         set_weights(ifunc(dims.rows, dims.cols));
      }
   }

   template<typename _TLAYER>
   inline
   void
   NetworkLayerImpl<_TLAYER>::set_input_size(size_t _rawin_sz)
   {
      _TLAYER::resize_input(_rawin_sz);
      this->layer_state.resize(this->size(), _rawin_sz);
   }
}

#endif //FLEX_NEURALNET_NETWORKLAYERIMPL_H_
