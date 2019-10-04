//
// Created by kfedrick on 6/7/19.
//

#ifndef FLEX_NEURALNET_EXTERNALINPUTRECORD_H_
#define FLEX_NEURALNET_EXTERNALINPUTRECORD_H_

#include <stddef.h>
#include <string>

namespace flexnnet
{
   class ExternalInputRecord
   {
   public:
      ExternalInputRecord();
      ExternalInputRecord(const std::string& _field, size_t _sz, size_t _index);

      const std::string& get_field(void) const;
      size_t get_size(void) const;
      size_t get_index(void) const;

   private:
      std::string field;
      size_t size;
      size_t index;
   };

   inline const std::string& ExternalInputRecord::get_field(void) const
   {
      return field;
   }

   inline size_t ExternalInputRecord::get_size(void) const
   {
      return size;
   }

   inline size_t ExternalInputRecord::get_index(void) const
   {
      return index;
   }
}

#endif //_EXTERNALINPUTRECORD_H_
