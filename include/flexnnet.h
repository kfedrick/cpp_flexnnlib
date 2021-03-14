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
   //typedef std::map<std::string, std::valarray<double>> NNetIO_Map;
   typedef  std::map<std::string, flexnnet::LayerWeights> NetworkWeights;

/*
   class NNetIO_Map : public std::map<std::string, std::valarray<double>>
   {
   public:

      NNetIO_Map() {}
      NNetIO_Map(const std::map<std::string, std::valarray<double>>& _map)
      {

      }

      const ValarrMap& value_map(void) const
      {
         return *this;
      }

      void encode(const flexnnet::ValarrMap& _vmap)
      {
      };

   };
*/

   // Alias declaration for Exemplar (e.g. Exemplar<valarray<double>,valarray<double>>)
   template <typename _InTyp, typename _OutTyp>
   using Exemplar = std::pair<_InTyp, _OutTyp>;
}

#include <flexnnet_utils.h>
#include <flexnnet_layers.h>
#include <flexnnet_networks.h>
#include <flexnnet_trainers.h>



#endif /* FLEX_NEURALNET_H_ */
