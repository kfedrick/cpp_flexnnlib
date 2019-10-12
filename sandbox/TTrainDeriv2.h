//
// Created by kfedrick on 9/15/19.
//

#ifndef _TTRAINDERIV2_H_
#define _TTRAINDERIV2_H_

#include "TTrain.h"
#include "TTrainDeriv.h"

//template<class _A, class _B, template<typename _X, typename _Y> class _C>
template<class _A, class _B, template<class = _A, class =_B> class _C>

//template<class _A, class _B, template<class _C> class _E>
//template<class _A, class _B, TTrainDeriv<_A, _B, class _C>>

class TTrainDeriv2
{
public:
   TTrainDeriv2(_A _a, _B _b);
   TTrainDeriv2(_A _a, _B _b, _C<_A, _B>* _c);
   _A get_A();
   _B get_B();

private:
   _C<_A, _B> c;
};

template<class _A, class _B, template<typename _X, typename _Y> class _C>
TTrainDeriv2<_A, _B, _C>::TTrainDeriv2(_A _a, _B _b)
{
}

template<class _A, class _B, template<typename _X, typename _Y> class _C>
TTrainDeriv2<_A, _B, _C>::TTrainDeriv2(_A _a, _B _b, _C<_A, _B>* _c)
{
   c = _C<_A, _B>(_a, _b);
}

template<class _A, class _B, template<typename _X, typename _Y> class _C> _A TTrainDeriv2<_A, _B, _C>::get_A()
{
   return c.get_A();
}

template<class _A, class _B, template<typename _X, typename _Y> class _C> _B TTrainDeriv2<_A, _B, _C>::get_B()
{
   return c.get_B();
}

#endif //_TTRAINDERIV2_H_
