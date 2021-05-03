/*
 * neuralnet.h
 *
 *  Created on: Jan 31, 2014
 *      Author: kfedrick
 */

#ifndef FLEX_NEURALNET_H_
#define FLEX_NEURALNET_H_

#include <map>
#include <valarray>
#include <LayerWeights.h>
#include <LayerState.h>

namespace flexnnet
{
/*
 * Global type definitions
 */
   typedef std::map<std::string, std::valarray<double>> ValarrMap;
   typedef unsigned int _index_typ;
   typedef std::map<std::string, flexnnet::LayerWeights> NetworkWeights;
   typedef std::map<std::string, flexnnet::LayerState> NetworkState;

   typedef std::tuple<double, double> EvalResults;
}

#include <flexnnet_utils.h>
#include <flexnnet_layers.h>
#include <flexnnet_networks.h>
#include <flexnnet_trainers.h>



#endif /* FLEX_NEURALNET_H_ */
