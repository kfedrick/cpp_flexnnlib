//
// Created by kfedrick on 5/18/21.
//

#ifndef FLEX_NEURALNET_VARIADICNETWORKINPUT_H_
#define FLEX_NEURALNET_VARIADICNETWORKINPUT_H_

#include <NetworkInput.h>

namespace flexnnet
{
   template<typename ...Features>
   class VariadicNetworkInput : public NetworkInput
   {
   public:
      VariadicNetworkInput();
      VariadicNetworkInput(const std::tuple<Features...>& _features);
      VariadicNetworkInput(const Features&... _features);
      VariadicNetworkInput(const std::pair<std::string, Features>&... _features);

      int count() const;
      void vectorize();
      void vectorize2();
      void set(const std::tuple<Features...>& _features);
      void set(const Features&... _features);
      void set(const std::pair<std::string, Features>&... _features);

      virtual const ValarrMap& value_map() const;

   protected:

      /*
       * Vectorize helper functions
       */
      template<std::size_t I=0>
      typename std::enable_if<I == sizeof...(Features), void>::type
      vectorize(const std::tuple<Features...>& t);

      template<std::size_t I>
      typename std::enable_if<I < sizeof...(Features), void>::type
      vectorize(const std::tuple<Features...>& t);

      template<std::size_t I=0>
      typename std::enable_if<I == sizeof...(Features), void>::type
      vectorize2(const std::tuple<std::pair<std::string,Features>...>& t);

      template<std::size_t I>
      typename std::enable_if<I < sizeof...(Features), void>::type
      vectorize2(const std::tuple<std::pair<std::string,Features>...>& t);

      /*
       * Assignment helper functions
       */
      template<std::size_t I=0>
      typename std::enable_if<I == sizeof...(Features), void>::type
      set(const std::tuple<Features...>& t);

      template<std::size_t I>
      typename std::enable_if<I < sizeof...(Features), void>::type
      set(const std::tuple<Features...>& t);

      template<std::size_t I, typename F, typename ...R>
      void set(const F& _first, const R&... _rem);

      template<std::size_t I, typename ...R>
      void
      set();

      template<std::size_t I, typename F, typename ...R>
      void alloc(const std::pair<std::string, F>& _first, const std::pair<std::string, R>&... _rem);

      template<std::size_t I, typename ...R>
      void alloc();

   private:
      std::tuple<Features...> raw_features;
      std::tuple<std::pair<std::string,Features>...> labeled_features;
      std::map<std::string, std::valarray<double>> feature_vectors;
   };

   template<typename ...Fs>
   VariadicNetworkInput<Fs...>::VariadicNetworkInput()
   {
   }

   template<typename ...Fs>
   VariadicNetworkInput<Fs...>::VariadicNetworkInput(const std::tuple<Fs...>& _fs)
   {
      set<0>(_fs);
   }

   template<typename ...Fs>
   VariadicNetworkInput<Fs...>::VariadicNetworkInput(const Fs&... _fs)
   {
      set<0>(_fs...);
   }

   template<typename ...Fs>
   VariadicNetworkInput<Fs...>::VariadicNetworkInput(const std::pair<std::string, Fs>&... _fs)
   {
      alloc<0>(_fs...);
   }

   template<typename ...Fs>
   void
   VariadicNetworkInput<Fs...>::set(const std::tuple<Fs...>& _fs)
   {
      set<0>(_fs);
   }

   template<typename ...Fs>
   void VariadicNetworkInput<Fs...>::set(const Fs&... _fs)
   {
      set<0>(_fs...);
   }

   template<typename ...Fs>
   int VariadicNetworkInput<Fs...>::count() const
   {
      return sizeof...(Fs);
   }

   template<typename ...Fs>
   void VariadicNetworkInput<Fs...>::vectorize()
   {
      vectorize<0>(raw_features);
   }

   template<typename ...Fs>
   void VariadicNetworkInput<Fs...>::vectorize2()
   {
      vectorize2<0>(labeled_features);
   }

   template<typename ...Fs>
   const ValarrMap& VariadicNetworkInput<Fs...>::value_map() const
   {
      return feature_vectors;
   }

   /*
    * Vectorize helpers
    */
   template<typename ...Fs>
   template<std::size_t I>
   typename std::enable_if<I == sizeof...(Fs), void>::type
   VariadicNetworkInput<Fs...>::vectorize(const std::tuple<Fs...>& t)
   { }

   template<typename ...Fs>
   template<std::size_t I>
   typename std::enable_if<I < sizeof...(Fs), void>::type
   VariadicNetworkInput<Fs...>::vectorize(const std::tuple<Fs...>& t)
   {
      std::cout << std::get<I>(t) << std::endl;
      vectorize<I + 1>(t);
   }

   template<typename ...Fs>
   template<std::size_t I>
   typename std::enable_if<I == sizeof...(Fs), void>::type
   VariadicNetworkInput<Fs...>::vectorize2(const std::tuple<std::pair<std::string,Fs>...>& t)
   { }

   template<typename ...Fs>
   template<std::size_t I>
   typename std::enable_if<I < sizeof...(Fs), void>::type
   VariadicNetworkInput<Fs...>::vectorize2(const std::tuple<std::pair<std::string,Fs>...>& t)
   {
      std::cout << "label = " << std::get<I>(t).first << " " << std::get<I>(t).second << std::endl;
      vectorize2<I + 1>(t);
   }

   /*
    * Set from tuple helpers
    */
   template<typename ...Fs>
   template<std::size_t I>
   typename std::enable_if<I == sizeof...(Fs), void>::type
   VariadicNetworkInput<Fs...>::set(const std::tuple<Fs...>& t)
   { }

   template<typename ...Fs>
   template<std::size_t I>
   typename std::enable_if<I < sizeof...(Fs), void>::type
   VariadicNetworkInput<Fs...>::set(const std::tuple<Fs...>& t)
   {
      std::get<I>(raw_features) = std::get<I>(t);
      set<I + 1>(t);
   }

   /*
    * Set from variadic arguments helpers
    */
   template<typename ...Fs>
   template<std::size_t I, typename F, typename ...R>
   void
   VariadicNetworkInput<Fs...>::set(const F& _first, const R&... _rem)
   {
      std::get<I>(raw_features) = _first;
      set<I+1,R...>(_rem...);
   }

   template<typename ...Fs>
   template<std::size_t I, typename ...R>
   void
   VariadicNetworkInput<Fs...>::set()
   {
   }

   template<typename ...Fs>
   template<std::size_t I, typename F, typename ...R>
   void VariadicNetworkInput<Fs...>::alloc(const std::pair<std::string, F>& _first, const std::pair<std::string, R>&... _rem)
   {
      std::cout << "allocate " << _first.first << "\n";
      std::get<I>(raw_features) = _first.second;
      std::get<I>(labeled_features) = _first;
      alloc<I + 1, R...>(_rem...);
   }

   template<typename ...Fs>
   template<std::size_t I, typename ...R>
   void
   VariadicNetworkInput<Fs...>::alloc()
   {
   }

} // end namespace flexnnet

#endif //FLEX_NEURALNET_VARIADICNETWORKINPUT_H_
