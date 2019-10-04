//
// Created by kfedrick on 9/15/19.
//

#ifndef _TTRAIN_H_
#define _TTRAIN_H_


template<class _A, class _B>
class TTrain
{
public:
   TTrain();
   TTrain(_A _a, _B _b);

   _A get_A();
   _B get_B();

private:
   _A a;
   _B b;
};

template <class _A, class _B> TTrain<_A, _B>::TTrain()
{}

template <class _A, class _B> TTrain<_A, _B>::TTrain(_A _a, _B _b)
{
   a = _a;
   b = _b;
}

template <class _A, class _B> _A TTrain<_A, _B>::get_A()
{
   return a;
}

template <class _A, class _B> _B TTrain<_A, _B>::get_B()
{
   return b;
}

#endif //_TTRAIN_H_
