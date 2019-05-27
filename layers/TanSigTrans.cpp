//
// Created by kfedrick on 5/25/19.
//

#include "TanSigTrans.h"

#include <cmath>

flexnnet::TanSigTrans::TanSigTrans(unsigned int _sz) : MyNetSum(_sz)
{
   dAdN.resize(_sz, _sz);
}

flexnnet::TanSigTrans::~TanSigTrans()
{
}

void flexnnet::TanSigTrans::calc_layer_output (std::vector<double>& _out, const std::vector<double>& _netin)
{
   for (unsigned int i = 0; i < _out.size (); i++)
      _out[i] = 2.0 / (1.0 + exp (-2.0 * gain * (_netin[i]))) - 1.0;
}

const flexnnet::Array<double>& flexnnet::TanSigTrans::calc_dAdN(const std::vector<double>& _out, const std::vector<double>& _netin)
{
   dAdN = 0;
   for (unsigned int i = 0; i < _out.size (); i++)
      dAdN[i][i] = gain * (1 - _out[i] * _out[i]);

   return dAdN;
}