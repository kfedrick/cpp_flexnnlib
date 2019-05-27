//
// Created by kfedrick on 5/24/19.
//

#include "RadialBasisTrans.h"

#include <cmath>

flexnnet::RadialBasisTrans::RadialBasisTrans(unsigned int _sz) : MyEuclideanDist(_sz)
{
   dAdN.resize(_sz, _sz);
}

flexnnet::RadialBasisTrans::~RadialBasisTrans()
{
}

void flexnnet::RadialBasisTrans::calc_layer_output (std::vector<double>& _out, const std::vector<double>& _netin)
{
   for (unsigned int i = 0; i < _out.size (); i++)
      _out[i] = exp (-_netin[i]);
}

const flexnnet::Array<double>& flexnnet::RadialBasisTrans::calc_dAdN(const std::vector<double>& _out, const std::vector<double>& _netin)
{
   dAdN = 0;
   for (unsigned int i = 0; i < _out.size (); i++)
      dAdN[i][i] = -_out[i];

   return dAdN;
}