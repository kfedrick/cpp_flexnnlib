//
// Created by kfedrick on 5/19/19.
//

#include "LogSig.h"

#include <cmath>

using flexnnet::Array2D;
using flexnnet::BasicLayer;
using flexnnet::LogSig;

const LogSig::Parameters LogSig::DEFAULT_PARAMS = {.gain=1.0};

LogSig::LogSig(size_t _sz, const std::string& _name, const Parameters& _params)
   : NetSumLayer(_sz, _name)
{
   set_params(_params);
}

LogSig::LogSig(const LogSig& _logsig) : NetSumLayer(_logsig)
{
   copy(_logsig);
}

LogSig::~LogSig()
{
}

LogSig&
LogSig::operator=(const LogSig& _logsig)
{
   return *this;
}

std::shared_ptr<BasicLayer>
LogSig::clone(void) const
{
   std::shared_ptr<LogSig> clone = std::shared_ptr<LogSig>(new LogSig(*this));
   return clone;
}

void
LogSig::copy(const LogSig& _logsig)
{
   params = _logsig.params;
}

void
LogSig::calc_layer_output(const std::valarray<double>& _netinv, std::valarray<double>& _layerval)
{
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      _layerval[i] = 1.0 / (1.0 + exp(-params.gain * _netinv[i]));
}

void
LogSig::calc_dy_dnet(const std::valarray<double>& _outv, Array2D<double>& _dydnet)
{
   _dydnet = 0;
   for (size_t i = 0; i < const_layer_output_size_ref; i++)
      _dydnet.at(i, i) = params.gain * _outv[i] * (1 - _outv[i]);
}