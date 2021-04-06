//
// Created by kfedrick on 2/17/21.
//
#include <iostream>
#include "test/include/CommonTestFixtureFunctions.h"

using std::cout;
using std::valarray;
using flexnnet::Array2D;

bool
CommonTestFixtureFunctions::valarray_double_near(const std::valarray<double>& _target, const std::valarray<double>& _test, double _epsilon)
{
   if (_target.size() != _test.size())
      return false;

   for (unsigned int i = 0; i < _target.size(); i++)
      if (fabs(_target[i] - _test[i]) > _epsilon)
         return false;

   return true;
}

bool
CommonTestFixtureFunctions::vector_double_near(const std::valarray<double>& _target, const std::valarray<double>& _test, double _epsilon)
{
   if (_target.size() != _test.size())
      return false;

   for (unsigned int i = 0; i < _target.size(); i++)
      if (fabs(_target[i] - _test[i]) > _epsilon)
         return false;

   return true;
}

bool
CommonTestFixtureFunctions::array_double_near(const flexnnet::Array2D<double>& _target, const flexnnet::Array2D<double>& _test, double _epsilon)
{
   Array2D<double>::Dimensions dim = _test.size();
   Array2D<double>::Dimensions xdim = _target.size();

   if (dim.rows != xdim.rows || dim.cols != xdim.cols)
      return false;

   for (size_t i = 0; i < xdim.rows; i++)
      for (size_t j = 0; j < xdim.cols; j++)
         if (fabs(_target.at(i, j) - _test.at(i, j)) > _epsilon)
            return false;

   return true;
}



std::string CommonTestFixtureFunctions::prettyPrintVector(const std::string& _label, const valarray<double>& _vec, int _prec)
{
   std::stringstream ssout;
   ssout.precision(_prec);

   bool first = true;
   ssout << "\n\"" << _label << "\" : \n";
   ssout << "   [";
   for (auto& val : _vec)
   {
      if (!first)
         ssout << ", ";
      else
         first = false;

      ssout << val;
   }
   ssout << "]";

   return ssout.str();
}

std::string
CommonTestFixtureFunctions::prettyPrintArray(const std::string& _label, const flexnnet::Array2D<double>& _arr, int _prec)
{
   std::stringstream ssout;
   ssout.precision(_prec);

   bool first_col = true;
   bool first_row = true;
   ssout << "\n\"" << _label << "\" : \n";
   ssout << "   [\n";
   Array2D<double>::Dimensions wdim = _arr.size();
   for (unsigned int i = 0; i < wdim.rows; i++)
   {
      if (!first_row)
         ssout << "],\n";
      else
         first_row = false;

      ssout << "      [";

      for (unsigned int j = 0; j < wdim.cols; j++)
      {
         if (!first_col)
            ssout << ", ";
         else
            first_col = false;
         ssout << _arr.at(i, j);
      }

      first_col = true;
   }
   // end row
   ssout << "]\n";

   // end array
   ssout << "   ]";

   return ssout.str();
}

std::string CommonTestFixtureFunctions::printResults(const flexnnet::BasicLayer& _layer, int _prec)
{
   std::stringstream ssout;
   ssout.setf(std::ios::fixed, std::ios::floatfield);
   ssout.precision(_prec);

   ssout << prettyPrintVector("output", _layer(), _prec) << "\n";
   ssout << prettyPrintArray("dy_dnet", _layer.get_dy_dnet(), _prec) << "\n";
   ssout << prettyPrintArray("dnet_dw", _layer.get_dnet_dw(), _prec) << "\n";
   ssout << prettyPrintArray("dnet_dx", _layer.get_dnet_dx(), _prec) << "\n";

   cout << ssout.str();
   return ssout.str();
}

flexnnet::ValarrMap CommonTestFixtureFunctions::parse_datum(const rapidjson::Value& _obj)
{
   flexnnet::ValarrMap datum_fields;
   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      std::string field = _obj[i]["field"].GetString();
      size_t field_sz = _obj[i]["size"].GetUint64();
      size_t field_index = _obj[i]["index"].GetUint64();

      datum_fields[field] = std::valarray<double>(field_sz);

      const rapidjson::Value& vec = _obj[i]["value"];
      for (rapidjson::SizeType i = 0; i < vec.Size(); i++)
         datum_fields[field][i] = vec[i].GetDouble();
   }

   return datum_fields;
}

Array2D<double> CommonTestFixtureFunctions::parse_weights(const rapidjson::Value& _obj, size_t _rows, size_t _cols)
{
   Array2D<double> weights(_rows, _cols);

   for (rapidjson::SizeType i = 0; i < _obj.Size(); i++)
   {
      const rapidjson::Value& myrow = _obj[i];

      for (rapidjson::SizeType j = 0; j < myrow.Size(); j++)
         weights.at(i, j) = myrow[j].GetDouble();
   }

   return weights;
}