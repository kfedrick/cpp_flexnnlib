//
// Created by kfedrick on 5/18/21.
//

#ifndef FLEX_NEURALNET_NETWORKREINFORCEMENT_H_
#define FLEX_NEURALNET_NETWORKREINFORCEMENT_H_

#include <NetworkInput.h>
#include <NetworkOutput.h>
#include <list>

namespace flexnnet
{
   template<unsigned int N>
   class NetworkReinforcement : public Reinforcement, public NetworkOutput
   {
   public:
      NetworkReinforcement();
      NetworkReinforcement(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers);

      NetworkReinforcement<N>& operator=(const NetworkReinforcement<N>& _reinf);

      /* *****************************************************************
       * Interface methods for network reinforcement.
       */
      size_t size() const override;

      virtual const double& operator[](size_t _ndx) const;
      virtual const double& at(size_t _ndx) const;
      virtual const double& at(const std::string& _field) const;

      virtual const std::vector<std::string>& get_fields() const override;
      virtual const std::valarray<double>& value() const override;

   protected:
      void copy(const NetworkReinforcement& _reinf);
   };

   template<unsigned int N>
   inline
   NetworkReinforcement<N>::NetworkReinforcement() : NetworkOutput()
   {
   }

   template<unsigned int N>
   inline
   NetworkReinforcement<N>::NetworkReinforcement(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers)
      : NetworkOutput(_olayers)
   {
      // Validate that number of output layers matches the specified number of
      // reinforcement signals, N.
      if (_olayers.size() != N)
      {
         static std::stringstream sout;
         sout << "Error : NetworkReinforcement::NetworkReinforcement(_olayers) - "
              << "# layers (" << _olayers.size() << " != expected size<" << N << ").\n";
         throw std::invalid_argument(sout.str());
      }

      // Validate that the size of each output layer vector is 1, a scalar.
      for (auto a_layer : _olayers)
      {
         if (a_layer->size() != 1)
         {
            static std::stringstream sout;
            sout << "Error : NetworkReinforcement::NetworkReinforcement(_olayers) - "
                 << "output layer [" << a_layer->name() << "] size (" << a_layer->size()
                 << ") != expected size (1).\n";
            throw std::invalid_argument(sout.str());
         }
      }
   }

   template<unsigned int N>
   inline
   size_t NetworkReinforcement<N>::size() const
   {
      return NetworkOutput::size();
   }


   template<unsigned int N>
   inline
   NetworkReinforcement<N>& NetworkReinforcement<N>::operator=(const NetworkReinforcement<N>& _reinf)
   {
      NetworkOutput::copy(_reinf);
      return *this;
   }

   template<unsigned int N>
   inline
   void NetworkReinforcement<N>::copy(const NetworkReinforcement& _reinf)
   {
      NetworkOutput::copy(_reinf);
   }

   template<unsigned int N>
   inline
   const double&
   NetworkReinforcement<N>::operator[](size_t _ndx) const
   {
      return ((NetworkOutput&)(*this))[_ndx][0];
   }

   template<unsigned int N>
   inline
   const double&
   NetworkReinforcement<N>::at(size_t _ndx) const
   {
      return ((NetworkOutput&)(*this)).at(_ndx)[0];
   }

   template<unsigned int N>
   inline
   const double& NetworkReinforcement<N>::at(const std::string& _field) const
   {
      return ((NetworkOutput&)(*this)).at(_field)[0];
   }

   template<unsigned int N>
   inline
   const std::vector<std::string>& NetworkReinforcement<N>::get_fields() const
   {
      return ((NetworkOutput*) this)->get_fields();
   }

   template<unsigned int N>
   inline
   const std::valarray<double>& NetworkReinforcement<N>::value() const
   {
      return ((NetworkOutput*) this)->value();
   }

} // end namespace flexnnet

#endif //FLEX_NEURALNET_NETWORKREINFORCEMENT_H_
