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

void flexnnet::JSONEncoder::ArrayToJSONObj (rapidjson::Value &_val, const Array2D<double> &_arr)
{
   Array2D<double>::Dimensions dim = _arr.size();

   _val.SetArray ();
   for (int i = 0; i < dim.rows; i++)
   {
      rapidjson::Value myrow (rapidjson::kArrayType);
      for (int j = 0; j < dim.cols; j++)
      {
         myrow.PushBack (_arr.at(i, j), allocator);
      }
      _val.PushBack (myrow, allocator);
   }
}

void flexnnet::JSONEncoder::JSONObjToArray (flexnnet::Array2D<double> &_arr, const rapidjson::Value &_obj)
{
   Array2D<double>::Dimensions dim = _arr.size();

   for (rapidjson::SizeType i=0; i<_obj.Size(); i++)
   {
      const rapidjson::Value& myrow = _obj[i];

      for (rapidjson::SizeType j=0; j<myrow.Size(); j++)
         _arr.at(i, j) = myrow[j].GetDouble();
   }
}