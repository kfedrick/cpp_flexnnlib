//
// Created by kfedrick on 5/19/19.
//

#include "LogSigTrans.h"

#include <cmath>

flexnnet::LogSigTrans::LogSigTrans(unsigned int _sz) : MyNetSum(_sz)
{
   dAdN.resize(_sz, _sz);
}

flexnnet::LogSigTrans::~LogSigTrans()
{
}

void flexnnet::LogSigTrans::calc_layer_output (std::vector<double>& _out, const std::vector<double>& _netin)
{
   for (unsigned int i = 0; i < _out.size (); i++)
      _out[i] = 1.0 / (1.0 + exp (-gain * _netin[i]));
}

const flexnnet::Array<double>& flexnnet::LogSigTrans::calc_dAdN(const std::vector<double>& _out, const std::vector<double>& _netin)
{
   dAdN = 0;
   for (unsigned int i = 0; i < _out.size (); i++)
      dAdN[i][i] = gain * _out[i] * (1 - _out[i]);

   return dAdN;
}