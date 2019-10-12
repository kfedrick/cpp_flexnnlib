//
// Created by kfedrick on 9/15/19.
//

#ifndef _TTRAINDERIV_H_
#define _TTRAINDERIV_H_

#include "TTrain.h"

template<class _A, class _B, class _C>
class TTrainDeriv : public TTrain<_A, _B>
{
public:
   TTrainDeriv(_A _a, _B _b, _C _c);
   _A get_A();
   _C get_C();

private:
   _C c;
};

template<class _A, class _B, class _C>
TTrainDeriv<_A, _B, _C>::TTrainDeriv(_A _a, _B _b, _C _c) : TTrain<_A, _B>(_a, _b)
{
   c = _c;
}

template<class _A, class _B, class _C> _A TTrainDeriv<_A, _B, _C>::get_A()
{
   return TTrain<_A, _B>::get_A() * 2;
}

template<class _A, class _B, class _C> _C TTrainDeriv<_A, _B, _C>::get_C()
{
   return c;
}

#endif //_TTRAINDERIV_H_
