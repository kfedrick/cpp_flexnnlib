//
// Created by kfedrick on 5/22/21.
//

#ifndef FLEX_NEURALNET_FEATURESET_H_
#define FLEX_NEURALNET_FEATURESET_H_

#include <type_traits>
#include <flexnnet.h>
#include <Feature.h>

namespace flexnnet
{
   template<typename Fs> class FeatureSet;
}

// Forward declarations for stream operators
template<typename Fs>
std::ostream& operator<<(std::ostream& _ostrm, const flexnnet::FeatureSet<Fs>& _featureset);

template<typename Fs>
std::istream& operator>>(std::istream& _istrm, flexnnet::FeatureSet<Fs>& _featureset);

namespace flexnnet
{
   template<typename Fs> class FeatureSet
   {
   public:
      static const size_t SIZE = std::tuple_size<Fs>::value;

   private:
      static std::array<std::string, SIZE> feature_names;

   public:
      FeatureSet();
      FeatureSet(const std::array<std::string, SIZE>& _names);
      FeatureSet(const FeatureSet<Fs>& _fs);

      FeatureSet& operator=(const FeatureSet<Fs>& _fs);

      /**
       * Return features count
       *
       * @tparam Fs
       * @return
       */
      size_t size() const;

      /**
       *
       * @return
       */
      const std::array<std::string, SIZE>& get_feature_names() const;

      Feature& operator[](size_t _ndx);
      const Feature& operator[](size_t _ndx) const;
      Feature& at(const std::string& _id);
      const Feature& at(const std::string& _id) const;

      size_t get_feature_index(const std::string& _id) const;

      /**
       * Return const tuple of features
       * @tparam Fs
       * @return
       */
      virtual const Fs& get_features() const;

      /**
       * Return tuple containing features
       *
       * @tparam Fs
       * @return
       */
      virtual Fs& get_features();

      virtual const ValarrMap& value_map() const;


      /**
       * Decode all features using the specified vector encodings.
       *
       * @tparam Fs
       * @param _encodings
       */
      virtual void decode(const std::vector<std::valarray<double>>& _encodings);

   protected:
      const std::vector<Feature*>& get_feature_pointers() const;

      template<size_t I = 0>
      typename std::enable_if<I < SIZE, void>::type
      decode_helper(const std::vector<std::valarray<double>>& _encodings);

      template<size_t I>
      typename std::enable_if<I == SIZE, void>::type
      decode_helper(const std::vector<std::valarray<double>>& _encodings) {};

      template<size_t I>
      typename std::enable_if<I < SIZE, void>::type
      set_feature_pointers();

      template<size_t I>
      typename std::enable_if<I == SIZE, void>::type
      set_feature_pointers() {};


   private:
      void initialize();
      void initialize_feature_names();
      void gather() const;

   private:
      Fs features;
      std::vector<Feature*> feature_ptrs;
      std::map<std::string, size_t> feature_indices;

      // TODO - change base NN activate() so I can remove this
      mutable ValarrMap vmap;
   };

   template<typename Fs>
   std::array<std::string, FeatureSet<Fs>::SIZE> FeatureSet<Fs>::feature_names;


   template<typename Fs> FeatureSet<Fs>::FeatureSet()
   {
      initialize();
      initialize_feature_names();
   }

   template<typename Fs> FeatureSet<Fs>::FeatureSet(const std::array<std::string, SIZE>& _feature_names)
   {
      initialize();

      // TODO - validate number of feature names == number of features
      feature_names = _feature_names;
      for (int ndx = 0; ndx<feature_names.size(); ndx++)
         feature_indices[feature_names[ndx]] = ndx;
   }

   template<typename Fs> FeatureSet<Fs>::FeatureSet(const FeatureSet<Fs>& _fs)
   {
      features = _fs.features;

      feature_ptrs.resize(SIZE);
      set_feature_pointers<0>();

      feature_names = _fs.feature_names;
      feature_indices = _fs.feature_indices;
   }

   template<typename Fs>
   inline
   FeatureSet<Fs>& FeatureSet<Fs>::operator=(const FeatureSet<Fs>& _fs)
   {
      features = _fs.features;

      feature_ptrs.resize(SIZE);
      set_feature_pointers<0>();

      feature_names = _fs.feature_names;
      feature_indices = _fs.feature_indices;
   }

   template<typename Fs>
   inline
   void FeatureSet<Fs>::initialize()
   {
      /*
       * Save feature count and set sizes of lists.
       */
      feature_ptrs.resize(SIZE);

      // Initialize feature names to defaults
      initialize_feature_names();

      // Set up the feature pointers
      set_feature_pointers<0>();
   }

   template<typename Fs>
   inline
   void FeatureSet<Fs>::initialize_feature_names()
   {
      std::stringstream featurename_ss;
      for (int ndx = 0; ndx < SIZE; ndx++)
      {
         featurename_ss.str(std::string());
         featurename_ss << "F" << ndx;

         feature_names[ndx] = featurename_ss.str();
         feature_indices[featurename_ss.str()] = ndx;
      }
   }

   template<typename Fs>
   inline
   size_t FeatureSet<Fs>::size() const
   {
      return SIZE;
   }

   template<typename Fs>
   inline
   const std::array<std::string, FeatureSet<Fs>::SIZE>& FeatureSet<Fs>::get_feature_names() const
   {
      return feature_names;
   }

   template<typename Fs>
   inline
   size_t FeatureSet<Fs>::get_feature_index(const std::string& _id) const
   {
      return feature_indices.at(_id);
   }

   template<typename Fs>
   inline
   Feature& FeatureSet<Fs>::operator[](size_t _ndx)
   {
      return *feature_ptrs[_ndx];
   }

   template<typename Fs>
   inline
   const Feature& FeatureSet<Fs>::operator[](size_t _ndx) const
   {
      return *feature_ptrs[_ndx];
   }

   template<typename Fs>
   inline
   Feature& FeatureSet<Fs>::at(const std::string& _id)
   {
      return *feature_ptrs[feature_indices.at(_id)];
   }

   template<typename Fs>
   inline
   const Feature& FeatureSet<Fs>::at(const std::string& _id) const
   {
      return *feature_ptrs[feature_indices.at(_id)];
   }

   template<typename Fs>
   inline
   const std::vector<Feature*>& FeatureSet<Fs>::get_feature_pointers() const
   {
      return feature_ptrs;
   }

   template<typename Fs>
   inline
   const Fs& FeatureSet<Fs>::get_features() const
   {
      return features;
   }

   template<typename Fs>
   inline
   Fs& FeatureSet<Fs>::get_features()
   {
      return features;
   }

   template<typename Fs>
   inline
   const ValarrMap& FeatureSet<Fs>::value_map() const
   {
      gather();
      return vmap;
   }

   template<typename Fs>
   inline
   void FeatureSet<Fs>::gather() const
   {
      size_t sz = size();
      for (int ndx = 0; ndx < sz; ndx++)
         vmap[feature_names[ndx]] = feature_ptrs[ndx]->get_encoding();
   }

   template<typename Fs>
   inline
   void FeatureSet<Fs>::decode(const std::vector<std::valarray<double>>& _encodings)
   {
      decode_helper<0>(_encodings);
   }

   template<typename Fs>
   template<size_t I>
   inline
   typename std::enable_if<I < FeatureSet<Fs>::SIZE, void>::type
   FeatureSet<Fs>::decode_helper(const std::vector<std::valarray<double>>& _encodings)
   {
      std::get<I>(features).decode(_encodings[I]);
      decode_helper<I + 1>(_encodings);
   }

   template<typename Fs>
   template<size_t I>
   typename std::enable_if<I < FeatureSet<Fs>::SIZE, void>::type
   FeatureSet<Fs>::set_feature_pointers()
   {
      //static_assert(std::is_base_of<Feature, std::declval(std::get<I>(features).())>::value,
      //              "FeatureSet<F,StateFs...> F must be derived from Feature");

      feature_ptrs[I] = &std::get<I>(features);
      set_feature_pointers<I + 1>();
   }
}

#endif // FLEX_NEURALNET_FEATURESET_H_
