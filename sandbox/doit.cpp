//
// Created by kfedrick on 9/15/19.
//

#include <iostream>
#include "TTrain.h"
#include "TTrainDeriv.h"
#include "TTrainDeriv2.h"

template<class, class> class X
{
};

int main(int argc, char** argv)
{
   TTrain<int, float> tt(1, 0.3);
   std::cout << tt.get_A() << " " << tt.get_B() << std::endl;

   TTrainDeriv<int, float, char> ttd(1, 0.3, 'a');
   std::cout << ttd.get_A() << " " << ttd.get_B() << " " << ttd.get_C() << std::endl;

   TTrain<int, float>& ttd_base = ttd;
   std::cout << ttd_base.get_A() << " " << ttd_base.get_B() << std::endl;

   TTrainDeriv2<int, float, TTrain> ttd2(1, 0.3, &tt);
   std::cout << ttd2.get_A() << " " << ttd2.get_B() << std::endl;

   return 0;
}

