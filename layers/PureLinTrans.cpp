//
// Created by kfedrick on 5/16/19.
//

#include "PureLinTrans.h"

flexnnet::PureLinTrans::PureLinTrans(unsigned int _sz) : MyNetSum(_sz)
{
   tranfersvec_size = _sz;
   dAdN.resize(_sz, _sz);
}

flexnnet::PureLinTrans::~PureLinTrans()
{
}

void
flexnnet::PureLinTrans::calc_layer_output (std::vector<double> &_out, const std::vector<double> &_netin)
{
   for (unsigned int i = 0; i < _out.size (); i++)
      _out[i] = _netin[i];
}

const flexnnet::Array<double>&
flexnnet::PureLinTrans::calc_dAdN (const std::vector<double> &_out, const std::vector<double> &_netin)
{
   dAdN = 0;
   for (unsigned int i = 0; i < _out.size (); i++)
      dAdN[i][i] = 1;

   return dAdN;
}

