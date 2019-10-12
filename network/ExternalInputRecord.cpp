//
// Created by kfedrick on 6/7/19.
//

#include "ExternalInputRecord.h"

using flexnnet::ExternalInputRecord;

ExternalInputRecord::ExternalInputRecord()
{}

ExternalInputRecord::ExternalInputRecord(const std::string& _field, size_t _sz, size_t _index)
{
   field = _field;
   size = _sz;
   index = _index;
}