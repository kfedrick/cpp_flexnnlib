//
// Created by kfedrick on 5/23/21.
//

#ifndef FLEX_NEURALNET_REINFORCEMENT_H_
#define FLEX_NEURALNET_REINFORCEMENT_H_

#include <RawFeatureSet.h>

namespace flexnnet
{
   template<size_t N=1>
   class Reinforcement : public RawFeatureSet<N>
   {
   public:
      Reinforcement();
      Reinforcement(const std::string& _name);
      Reinforcement(const Reinforcement& _r);

      void fill(double _val);
      virtual void set(size_t _ndx, double _val);

      virtual const Feature& operator[](size_t _ndx) const;
      virtual const Feature& at(const std::string& _field) const;
   };

   template<size_t N>
   inline
   Reinforcement<N>::Reinforcement() : RawFeatureSet<N>()
   {
   }

   template<size_t N>
   inline
   Reinforcement<N>::Reinforcement(const std::string& _name) : RawFeatureSet<N>({_name})
   {
   }

   template<size_t N>
   inline
   Reinforcement<N>::Reinforcement(const Reinforcement& _r) : RawFeatureSet<N>(_r)
   {

   }

   template<size_t N>
   inline
   const Feature& Reinforcement<N>::operator[](size_t _ndx) const
   {
      return std::get<0>(this->get_features());
      //return std::get<0>(this->get_features()).get_encoding()[_ndx];
   }

   template<size_t N>
   inline
   void Reinforcement<N>::fill(double _val)
   {
      std::valarray<double> v(_val,N);
      std::get<0>(this->get_features()).decode(v);
   }

   template<size_t N>
   inline
   void Reinforcement<N>::set(size_t _ndx, double _val)
   {
      // TODO - horribly inefficient, fix this
      std::valarray<double> v = std::get<0>(this->get_features()).get_encoding();
      v[_ndx] = _val;
      std::get<0>(this->get_features()).decode(v);
   }

/*   template<size_t N>
   inline
   double& Reinforcement<N>::operator[](size_t _ndx)
   {
      return std::get<0>(this->get_features()).value()[_ndx];
   }*/

   template<size_t N>
   inline
   const Feature& Reinforcement<N>::at(const std::string& _field) const
   {
      //return (*this)[0].get_encoding()[this->get_feature_index(_field)];
      //return std::get<0>(this->get_features()).value()[this->get_feature_index(_field)];
      return std::get<0>(this->get_features());

   }

}

#endif // FLEX_NEURALNET_REINFORCEMENT_H_
