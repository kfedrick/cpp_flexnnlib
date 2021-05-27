//
// Created by kfedrick on 5/18/21.
//

#ifndef FLEX_NEURALNET_VARIADICINPUT_H_
#define FLEX_NEURALNET_VARIADICINPUT_H_

#include <NetworkInput.h>

namespace flexnnet
{
   template<typename ...Features>
   class VariadicInput : public NetworkInput, public Vectorizable
   {
   public:
      static const size_t FEATURE_COUNT = sizeof...(Features);

   public:
      VariadicInput();
      VariadicInput(const std::pair<std::string, Features>&... _features);
      VariadicInput(const VariadicInput<Features...>& _vni);

      VariadicInput<Features...>& operator=(const VariadicInput<Features...>& _vni);

      int count() const;
      void set(const std::tuple<Features...>& _features);
      void set(const Features&... _features);

      const std::vector<std::string>& get_labels() const;
      const std::tuple<Features...>& values() const;
      std::tuple<Features...>& values();
      const std::valarray<double>& vectorize() const;
      virtual const ValarrMap& value_map() const;

   protected:

      void vectorize_features() const;
      void update_concatenated_size() const;
      void concatenate_values() const;

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

   private:
      void copy(const VariadicInput<Features...>& _vni);

      template<std::size_t I, typename F, typename ...R>
      void reset_feature_ptrs();

      template<std::size_t I>
      void reset_feature_ptrs();

      template<size_t I, typename F>
      void alloc_feature(const F& _f);

      template<size_t I>
      void alloc_feature(const std::valarray<double>& _f);

      template<size_t I, typename F>
      void set_feature_ptr(F& _f);

      template<size_t I>
      void set_feature_ptr(std::valarray<double>& _f);

      template<std::size_t I, typename F, typename ...R>
      void alloc(const std::pair<std::string, F>& _feature, const std::pair<std::string, R>&... _rem);

      template<std::size_t I, typename ...R>
      void alloc();

   private:
      std::vector<std::string> labels;
      std::tuple<Features...> raw_features;
      std::vector<Vectorizable*> vectorizable_feature_ptrs;
      std::map<std::string, size_t> feature_indices;

      mutable std::map<std::string, std::valarray<double>> feature_vectors;
      mutable std::valarray<double> concatenated_value_vector;

   };




   template<typename ...Fs>
   VariadicInput<Fs...>::VariadicInput()
   {
      labels.resize(FEATURE_COUNT);
      vectorizable_feature_ptrs.resize(FEATURE_COUNT);
   }

   template<typename ...Fs>
   VariadicInput<Fs...>::VariadicInput(const std::pair<std::string, Fs>&... _fs)
   {
      labels.resize(FEATURE_COUNT);
      vectorizable_feature_ptrs.resize(FEATURE_COUNT);
      alloc<0>(_fs...);
   }

   template<typename ...Fs>
   VariadicInput<Fs...>::VariadicInput(const VariadicInput<Fs...>& _vni)
   {
      vectorizable_feature_ptrs.resize(FEATURE_COUNT);
      copy(_vni);
   }

   template<typename ...Fs>
   void
   VariadicInput<Fs...>::copy(const VariadicInput<Fs...>& _vni)
   {
      labels = _vni.labels;
      raw_features = _vni.raw_features;
      feature_indices = _vni.feature_indices;
      feature_vectors = _vni.feature_vectors;

      reset_feature_ptrs<0, Fs...>();
   }

   template<typename ...Fs>
   VariadicInput<Fs...>& VariadicInput<Fs...>::operator=(const VariadicInput<Fs...>& _vni)
   {
      vectorizable_feature_ptrs.resize(FEATURE_COUNT);
      copy(_vni);
      return *this;
   }

   template<typename ...Fs>
   int VariadicInput<Fs...>::count() const
   {
      return sizeof...(Fs);
   }

   template<typename ...Fs>
   void
   VariadicInput<Fs...>::set(const std::tuple<Fs...>& _fs)
   {
      set<0>(_fs);
      reset_feature_ptrs<0>();
   }

   template<typename ...Fs>
   void VariadicInput<Fs...>::set(const Fs&... _fs)
   {
      set<0>(_fs...);
      reset_feature_ptrs<0>();
   }

   template<typename ...Fs>
   const std::vector<std::string>& VariadicInput<Fs...>::get_labels() const
   {
      return labels;
   }

   template<typename ...Fs>
   const std::tuple<Fs...>& VariadicInput<Fs...>::values() const
   {
      return raw_features;
   }

   template<typename ...Fs>
   std::tuple<Fs...>& VariadicInput<Fs...>::values()
   {
      return raw_features;
   }

   template<typename ...Fs>
   const ValarrMap& VariadicInput<Fs...>::value_map() const
   {
      vectorize_features();
      return feature_vectors;
   }

   template<typename ...Fs>
   const std::valarray<double>& VariadicInput<Fs...>::vectorize() const
   {
      vectorize_features();
      concatenate_values();
      return concatenated_value_vector;
   }

   template<typename ...Fs>
   void VariadicInput<Fs...>::concatenate_values() const
   {
      update_concatenated_size();

      unsigned int vndx = 0;
      for (auto a_vector : feature_vectors)
      {
         const std::valarray<double>& vec = a_vector.second;
         int vsz = vec.size();
         for (int ndx=0; ndx<vsz; ndx++)
            concatenated_value_vector[vndx++] = vec[ndx];
      }
   }

   template<typename ...Fs>
   void VariadicInput<Fs...>::update_concatenated_size() const
   {
      size_t cvec_sz = 0;
      for (auto a_fv : feature_vectors)
         cvec_sz += a_fv.second.size();

      if (concatenated_value_vector.size() != cvec_sz)
         concatenated_value_vector.resize(cvec_sz);
   }

   template<typename ...Fs>
   void VariadicInput<Fs...>::vectorize_features() const
   {
//      std::cout << "feature vector size " << feature_vectors.size() << "\n" << std::flush;
      for (int ndx=0; ndx < FEATURE_COUNT; ndx++)
         feature_vectors[labels[ndx]] = vectorizable_feature_ptrs[ndx]->vectorize();
   }

   /*
    * Set from tuple helpers
    */
   template<typename ...Fs>
   template<std::size_t I>
   typename std::enable_if<I == sizeof...(Fs), void>::type
   VariadicInput<Fs...>::set(const std::tuple<Fs...>& t)
   { }

   template<typename ...Fs>
   template<std::size_t I>
   typename std::enable_if<I < sizeof...(Fs), void>::type
   VariadicInput<Fs...>::set(const std::tuple<Fs...>& t)
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
   VariadicInput<Fs...>::set(const F& _first, const R&... _rem)
   {
      std::get<I>(raw_features) = _first;
      set<I+1,R...>(_rem...);
   }

   template<typename ...Fs>
   template<std::size_t I, typename ...R>
   void
   VariadicInput<Fs...>::set()
   {
   }

   template<typename ...Fs>
   template<std::size_t I, typename F, typename ...R>
   void VariadicInput<Fs...>::alloc(const std::pair<std::string, F>& _feature, const std::pair<std::string, R>&... _rem)
   {
      labels[I] = _feature.first;
      std::get<I>(raw_features) = _feature.second;
      set_feature_ptr<I>(std::get<I>(raw_features));
      feature_indices[_feature.first] = I;

      // recursive call to alloc
      alloc<I + 1, R...>(_rem...);
   }

   template<typename ...Fs>
   template<std::size_t I, typename ...R>
   void
   VariadicInput<Fs...>::alloc()
   {
   }

   template<typename ...Fs>
   template<std::size_t I, typename F, typename ...R>
   void VariadicInput<Fs...>::reset_feature_ptrs()
   {
      set_feature_ptr<I>(std::get<I>(raw_features));
      reset_feature_ptrs<I + 1, R...>();
   }

   template<typename ...Fs>
   template<std::size_t I>
   void
   VariadicInput<Fs...>::reset_feature_ptrs()
   {
   }

   template<typename ...Fs>
   template<size_t I, typename F>
   void VariadicInput<Fs...>::set_feature_ptr(F& _f)
   {
      vectorizable_feature_ptrs[I] = &_f;
   }

   template<typename ...Fs>
   template<size_t I>
   void VariadicInput<Fs...>::set_feature_ptr(std::valarray<double>& _f)
   {
      vectorizable_feature_ptrs[I] = new Vectorizable(std::get<I>(raw_features));
   }
} // end namespace flexnnet

#endif //FLEX_NEURALNET_VARIADICINPUT_H_
