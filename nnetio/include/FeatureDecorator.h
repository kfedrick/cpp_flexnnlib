//
// Created by kfedrick on 5/22/21.
//

#ifndef FLEX_NEURALNET_FEATUREDECORATOR_H_
#define FLEX_NEURALNET_FEATUREDECORATOR_H_

namespace flexnnet
{
   template<typename T>
   class FeatureDecorator : public T
   {
      static_assert(std::is_base_of<Feature, T>::value, "T must be derived from Feature");

   public:
      std::vector<std::string> ids = {"thing1", "thing2"};
      virtual void decode(const std::valarray<double>& _encoding);
      void doit();
   };

   template<typename T>
   void FeatureDecorator<T>::doit()
   {
      std::cout << "do it, do it, doit doit doit\n";
   }

   template<typename T>
   inline
   void FeatureDecorator<T>::decode(const std::valarray<double>& _encoding)
   {
      std::cout << "Feature Decorator decode\n";
      T::decode(_encoding);
   }
}

// end namespace

#endif // FLEX_NEURALNET_FEATUREDECORATOR_H_
