//
// Created by kfedrick on 5/16/19.
//

#include "JSONEncoder.h"

rapidjson::Document flexnnet::JSONEncoder::document;
rapidjson::Document::AllocatorType & flexnnet::JSONEncoder::allocator = flexnnet::JSONEncoder::document.GetAllocator();

void flexnnet::JSONEncoder::vectorToJSONObj (rapidjson::Value &_val, const std::vector<double> &_vec)
{
   _val.SetArray ();
   for (int i = 0; i < _vec.size (); i++)
      _val.PushBack (_vec[i], allocator);
}

void flexnnet::JSONEncoder::ArrayToJSONObj (rapidjson::Value &_val, const Array<double> &_arr)
{
   _val.SetArray ();
   for (int i = 0; i < _arr.rowDim (); i++)
   {
      rapidjson::Value myrow (rapidjson::kArrayType);
      for (int j = 0; j < _arr.colDim (); j++)
      {
         myrow.PushBack (_arr[i][j], allocator);
      }
      _val.PushBack (myrow, allocator);
   }
}

void flexnnet::JSONEncoder::JSONObjToArray (flexnnet::Array<double> &_arr, const rapidjson::Value &_obj)
{
   for (rapidjson::SizeType i=0; i<_obj.Size(); i++)
   {
      const rapidjson::Value& myrow = _obj[i];

      for (rapidjson::SizeType j=0; j<myrow.Size(); j++)
         _arr[i][j] = myrow[j].GetDouble();
   }
}