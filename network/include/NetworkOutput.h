//
// Created by kfedrick on 4/25/21.
//

#ifndef FLEX_NEURALNET_NETWORKOUTPUT_H_
#define FLEX_NEURALNET_NETWORKOUTPUT_H_

#include <flexnnet.h>
#include <Vectorizable.h>
#include <NetworkLayer.h>

namespace flexnnet
{
   class NetworkOutput : public Vectorizable
   {
   public:
      NetworkOutput();
      NetworkOutput(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers);
      NetworkOutput(const NetworkOutput& _nout);

      virtual size_t size() const;

      virtual NetworkOutput& operator=(const NetworkOutput& _nout);

      const std::valarray<double>& operator[](size_t _ndx) const;
      const std::valarray<double>& at(size_t _ndx) const;

      const std::valarray<double>& at(const std::string& _field) const;

      const std::vector<std::string>& get_field_names() const;
      const std::valarray<double>& vectorize() const;

      virtual void activate();
      void connect(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers);

   protected:
      void copy(const NetworkOutput& _nout);

      const std::vector<std::shared_ptr<NetworkLayer>>& get_output_layers() const;
      void update_concatenated_size() const;
      void concatenate_values() const;

   private:
      std::vector<std::string> fields;
      std::vector<std::shared_ptr<NetworkLayer>> output_layers;

      mutable bool stale_cvalue{true};
      mutable std::valarray<double> concatenated_value_vector;
      std::map<std::string, size_t> field_indices_map;
   };

   inline
   NetworkOutput::NetworkOutput()
   {
   }

   inline
   NetworkOutput::NetworkOutput(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers)
   {
      connect(_olayers);
   }

   inline
   void NetworkOutput::connect(const std::vector<std::shared_ptr<NetworkLayer>>& _olayers)
   {
      output_layers = _olayers;

      size_t ndx = 0;
      for (auto& a_layer : _olayers)
      {
         fields.push_back(a_layer->name());
         field_indices_map[a_layer->name()] = ndx++;
      }

      concatenate_values();
   }

   inline
   NetworkOutput::NetworkOutput(const NetworkOutput& _nout)
   {
      copy(_nout);
   }

   inline
   void NetworkOutput::copy(const NetworkOutput& _nout)
   {
      fields = _nout.fields;
      field_indices_map = _nout.field_indices_map;
      concatenated_value_vector = _nout.concatenated_value_vector;
      stale_cvalue = _nout.stale_cvalue;
      output_layers = _nout.output_layers;
   }

   inline
   size_t NetworkOutput::size() const
   {
      if (stale_cvalue)
      {
         concatenate_values();
         stale_cvalue = false;
      }
      return concatenated_value_vector.size();
   }

   inline
   NetworkOutput& NetworkOutput::operator=(const NetworkOutput& _nout)
   {
      copy(_nout);
      return *this;
   }

   inline
   const std::valarray<double>&
   NetworkOutput::operator[](size_t _ndx) const
   {
      return output_layers[_ndx]->value();
   }

   inline
   const std::valarray<double>&
   NetworkOutput::at(size_t _ndx) const
   {
      return output_layers.at(_ndx)->value();
   }

   inline
   const std::valarray<double>&
   NetworkOutput::at(const std::string& _field) const
   {
      return output_layers.at(field_indices_map.at(_field))->value();
   }

   inline
   const std::vector<std::shared_ptr<NetworkLayer>>& NetworkOutput::get_output_layers() const
   {
      return output_layers;
   }

   inline
   void NetworkOutput::activate()
   {
      stale_cvalue = true;
   }

   inline
   void NetworkOutput::update_concatenated_size() const
   {
      size_t cvec_sz = 0;
      for (auto a_layer : output_layers)
         cvec_sz += a_layer->value().size();

      if (concatenated_value_vector.size() != cvec_sz)
         concatenated_value_vector.resize(cvec_sz);
   }

   inline
   const std::vector<std::string>& NetworkOutput::get_field_names() const
   {
      return fields;
   }

   inline
   const std::valarray<double>& NetworkOutput::vectorize() const
   {
      if (stale_cvalue)
      {
         concatenate_values();
         stale_cvalue = false;
      }
      return concatenated_value_vector;
   }

   inline
   void NetworkOutput::concatenate_values() const
   {
      update_concatenated_size();

      unsigned int vndx = 0;
      for (auto a_layer : output_layers)
      {
         const std::valarray<double>& vec = a_layer->value();
         int vsz = vec.size();
         for (int ndx=0; ndx<vsz; ndx++)
            concatenated_value_vector[vndx++] = vec[ndx];
      }
   }
}
#endif //FLEX_NEURALNET_NETWORKOUTPUT_H_
