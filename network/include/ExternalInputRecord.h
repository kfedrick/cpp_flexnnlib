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
      ExternalInputRecord(const ExternalInputRecord& _rec);

      const std::string& field(void) const;
      size_t size(void) const;
      size_t index(void) const;

   private:
      std::string field_name;
      size_t field_size;
      size_t field_index;
   };

   inline const std::string& ExternalInputRecord::field(void) const
   {
      return field_name;
   }

   inline size_t ExternalInputRecord::size(void) const
   {
      return field_size;
   }

   inline size_t ExternalInputRecord::index(void) const
   {
      return field_index;
   }
}

#endif //_EXTERNALINPUTRECORD_H_
