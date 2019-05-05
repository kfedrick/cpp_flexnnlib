/*
 * Layer.h
 *
 *  Created on: Mar 9, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_LAYER_H_
#define FLEX_NEURALNET_LAYER_H_

#include "BaseLayer.h"

namespace flex_neuralnet
{

template <class _NetInFunc, class _TransFunc>
class Layer: public flex_neuralnet::BaseLayer
{
public:

   Layer(unsigned int sz, const char* _name = "Layer<_netInFunc, _TransFunc>");
   Layer(unsigned int sz, const string& _name);
   ~Layer();

   _TransFunc& get_transfer_functor();
   _NetInFunc& get_netinput_functor();

   operator BaseLayer*();

private:
   void initialize_policy();
};

template <class _NetInFunc, class _TransFunc>
Layer<_NetInFunc, _TransFunc>::Layer(unsigned int sz, const char* _name) : BaseLayer(sz, _name)
{
   initialize_policy();
}

template <class _NetInFunc, class _TransFunc>
Layer<_NetInFunc, _TransFunc>::Layer(unsigned int sz, const string& _name) : BaseLayer(sz, _name)
{
   initialize_policy();
}

template <class _NetInFunc, class _TransFunc>
Layer<_NetInFunc, _TransFunc>::~Layer()
{

}

template <class _NetInFunc, class _TransFunc> inline
void Layer<_NetInFunc, _TransFunc>::initialize_policy()
{
   set_netinput_functor(new _NetInFunc());
   set_transfer_functor(new _TransFunc());
}

template <class _NetInFunc, class _TransFunc> inline
_TransFunc& Layer<_NetInFunc, _TransFunc>::get_transfer_functor()
{
   BaseLayer* base_layer_ptr = dynamic_cast<BaseLayer*>(this);
   TransferFunctor* transfunc_ptr = base_layer_ptr->get_transfer_functor();
   return dynamic_cast<_TransFunc&>(*transfunc_ptr);
}

template <class _NetInFunc, class _TransFunc> inline
_NetInFunc& Layer<_NetInFunc, _TransFunc>::get_netinput_functor()
{
   BaseLayer* base_layer_ptr = dynamic_cast<BaseLayer*>(this);
   NetInputFunctor* netinfunc_ptr = base_layer_ptr->get_netinput_functor();
   return dynamic_cast<_NetInFunc&>(*netinfunc_ptr);
}

template <class _NetInFunc, class _TransFunc> inline
Layer<_NetInFunc, _TransFunc>::operator BaseLayer*()
{
   return dynamic_cast<BaseLayer>(this);
}

} /* namespace flex_neuralnet */

#endif /* FLEX_NEURALNET_LAYER_H_ */
