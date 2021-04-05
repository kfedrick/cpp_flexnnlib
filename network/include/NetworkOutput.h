//
// Created by kfedrick on 2/22/21.
//

#ifndef FLEX_NEURALNET_NETWORKOUTPUT_H_
#define FLEX_NEURALNET_NETWORKOUTPUT_H_

#include "flexnnet_networks.h"
#include "NetworkLayer.h"

namespace flexnnet
{
   class NetworkOutput : public NetworkLayer
   {
   public:
      NetworkOutput();
      ~NetworkOutput();

      const std::string& name() const override;

      size_t size() const override;

      const std::valarray<double>& value() const override;

      /**
       * Marshal layer inputs, activate the base layer and return the
       * layer output.
       * @param _externin
       * @return
       */
      const std::valarray<double>& activate(const ValarrMap& _externin) override;

   private:
      std::string layer_name;
   };


   inline
   const std::string& NetworkOutput::name() const
   {
      return layer_name;
   }

   inline
   size_t NetworkOutput::size() const
   {
      return virtual_input_vector_const_ref.size();
   }

   inline
   const std::valarray<double>& NetworkOutput::value() const
   {
      return virtual_input_vector_const_ref;
   }
}
#endif //FLEX_NEURALNET_NETWORKOUTPUT_H_
