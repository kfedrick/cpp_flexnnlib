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

namespace flexnnet
{
/*
 * Global type definitions
 */
   typedef std::map<std::string, std::valarray<double>> ValarrMap;
   typedef unsigned int _index_typ;
   typedef std::map<std::string, flexnnet::LayerWeights> NetworkWeights;

   typedef std::tuple<double, double> EvalResults;

   // Alias declaration for Exemplar and ExemplarSeries
   //template <typename _InTyp, typename _OutTyp>
   //using Exemplar = std::pair<_InTyp, _OutTyp>;

   //template <typename _InTyp, typename _OutTyp>
   //using ExemplarSeries = std::vector<std::pair<_InTyp, _OutTyp>>;

}

#include <flexnnet_utils.h>
#include <flexnnet_layers.h>
#include <flexnnet_networks.h>
#include <flexnnet_trainers.h>



#endif /* FLEX_NEURALNET_H_ */
