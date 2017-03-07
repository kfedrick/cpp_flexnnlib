/*
 * LogSig.cpp
 *
 *  Created on: Mar 10, 2014
 *      Author: kfedrick
 */

#include "LogSig.h"

namespace flex_neuralnet
{

LogSig::LogSig()
{
   gain = 1.0;
}

LogSig::~LogSig()
{
   // TODO Auto-generated destructor stub
}

double LogSig::get_gain() const
{
   return gain;
}

void LogSig::set_gain(double val)
{
   gain = val;
}


void LogSig::operator()(vector<double>& transVec, Array<double>& dAdN,
      Array<double>& dAdB, const vector<double>& netInVec, const vector<double>& biasVec) const
{
   for (unsigned int i=0; i<transVec.size(); i++)
      transVec[i] = 1.0 / (1.0 + exp( -gain * (biasVec[i] + netInVec[i]) ));

   dAdN = 0;
   for (unsigned int i=0; i<transVec.size(); i++)
      dAdN[i][i] = gain * transVec[i] * (1 - transVec[i]);

   dAdB = 0;
   for (unsigned int i=0; i<transVec.size(); i++)
      dAdB[i][i] = gain * transVec[i] * (1 - transVec[i]);
}

LogSig* LogSig::clone() const
{
   return new LogSig(*this);
}

} /* namespace flex_neuralnet */
