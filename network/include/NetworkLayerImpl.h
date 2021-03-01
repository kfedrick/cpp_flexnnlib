//
// Created by kfedrick on 2/21/21.
//

#ifndef FLEX_NEURALNET_NETWORKLAYERIMPL_H_
#define FLEX_NEURALNET_NETWORKLAYERIMPL_H_

#include <memory>
#include "flexnnet.h"
#include "BasicLayer.h"
#include "NetworkLayer.h"
#include "LayerConnRecord.h"

namespace flexnnet
{
   class NetworkTopology;

   class NetworkLayerImpl : public NetworkLayer
   {
      friend class NetworkTopology;

   public:
      NetworkLayerImpl();
      NetworkLayerImpl(bool _is_output);
      NetworkLayerImpl(const std::shared_ptr<BasicLayer>& _layer, bool _is_output = false);
      NetworkLayerImpl(std::shared_ptr<BasicLayer>&& _layer, bool _is_output = false);

      ~NetworkLayerImpl();

      /**
       * Marshal layer inputs, activate the base layer and return the
       * layer output.
       * @param _externin
       * @return
       */
      virtual const std::valarray<double>& activate(const NNetIO_Typ& _externin);


   protected:
      virtual std::shared_ptr<BasicLayer>& layer();

   };


   inline std::shared_ptr<BasicLayer>& NetworkLayerImpl::layer()
   {
      return basiclayer();
   }
}

#endif //FLEX_NEURALNET_NETWORKLAYER_H_
