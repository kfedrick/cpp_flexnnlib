//
// Created by kfedrick on 6/7/19.
//

#include "ExternalInputRecord.h"

using flexnnet::ExternalInputRecord;

ExternalInputRecord::ExternalInputRecord()
{}

ExternalInputRecord::ExternalInputRecord(const std::string& _field, size_t _sz, size_t _index)
{
   field_name = _field;
   field_size = _sz;
   field_index = _index;
}

ExternalInputRecord::ExternalInputRecord(const ExternalInputRecord& _rec)
{
   field_name = _rec.field_name;
   field_size = _rec.field_size;
   field_index = _rec.field_index;
}