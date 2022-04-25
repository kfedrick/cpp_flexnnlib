//
// Created by kfedrick on 5/22/21.
//

#ifndef FLEX_NEURALNET_NNFEATURESET_H_
#define FLEX_NEURALNET_NNFEATURESET_H_

#include <FeatureSetImpl.h>
#include <NetworkOutput.h>

namespace flexnnet
{
   /**
    * NNFeatureSet decorates FeatureSet and extends NetworkOutput
    */
   template<typename Fs>
   class NNFeatureSet : public Fs, public NetworkOutput
   {
   public:
      //static const size_t SIZE = std::tuple_size<Fs>::value;

      NNFeatureSet();
      NNFeatureSet(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers, const std::array<std::string, 1>& _lnames);

      NNFeatureSet(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers, const std::array<std::string, 2>& _lnames);
      virtual void activate() override;

      NNFeatureSet& operator=(const Fs _fs);
   };

   template<typename Fs>
   NNFeatureSet<Fs>::NNFeatureSet() : NetworkOutput()
   {
   }

   template<typename Fs>
   NNFeatureSet<Fs>::NNFeatureSet(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers, const std::array<std::string, 1>& _lnames) : Fs(_lnames), NetworkOutput(_olayers)
   {
      int i = 0;
      for (auto& it : _olayers)
         Fs::feature_names[i++] = it->name();
   }

   template<typename Fs>
   NNFeatureSet<Fs>::NNFeatureSet(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers, const std::array<std::string, 2>& _lnames) : Fs(_lnames), NetworkOutput(_olayers)
   {
      int i = 0;
      for (auto& it : _olayers)
         Fs::feature_names[i++] = it->name();
   }

   template<typename Fs>
   NNFeatureSet<Fs>& NNFeatureSet<Fs>::operator=(const Fs _fs)
   {
      Fs::operator=(_fs);
      return *this;
   }


   template<typename Fs>
   void NNFeatureSet<Fs>::activate()
   {
      NetworkOutput::activate();

      const std::vector<std::shared_ptr<NetworkLayer>>& olayers = get_output_layers();
      size_t layer_count = olayers.size();
      const std::vector<Feature*>& fptrs = Fs::get_feature_pointers();

      for (int i=0; i<layer_count; i++)
      {
         fptrs[i]->decode(olayers[i]->value());


/*         std::cout << "network feature set out:\n";
         std::valarray<double> va = olayers[i]->value();
         for (auto& val : va)
               std::cout << "  " << val << ", ";
         std::cout << "\n";*/

/*         std::valarray<double> vae = fptrs[i]->get_encoding();
         for (auto& val : vae)
            std::cout << "  " << val << ", ";
         std::cout << "\n";
         std::cout << "--\n";*/
      }

   }

} // end namespace



#endif // FLEX_NEURALNET_NNFEATURESET_H_
