//
// Created by kfedrick on 6/18/19.
//

#include "test/include/TestLayer.h"

using std::cout;
using std::valarray;
using flexnnet::Array2D;
using flexnnet::Datum;

bool
TestLayer::valarray_double_near (const std::valarray<double> &_target, const std::valarray<double> &_test, double _epsilon)
{
   if (_target.size () != _test.size ())
      return false;

   for (unsigned int i = 0; i < _target.size (); i++)
      if (fabs (_target[i] - _test[i]) > _epsilon)
         return false;

   return true;
}

bool
TestLayer::vector_double_near (const std::valarray<double> &_target, const std::valarray<double> &_test, double _epsilon)
{
   if (_target.size () != _test.size ())
      return false;

   for (unsigned int i = 0; i < _target.size (); i++)
      if (fabs (_target[i] - _test[i]) > _epsilon)
         return false;

   return true;
}

bool
TestLayer::array_double_near (const flexnnet::Array2D<double> &_target, const flexnnet::Array2D<double> &_test, double _epsilon)
{
   Array2D<double>::Dimensions dim = _test.size();
   Array2D<double>::Dimensions xdim = _target.size();

   if (dim.rows != xdim.rows || dim.cols != xdim.cols)
      return false;

   for (size_t i = 0; i < xdim.rows; i++)
      for (size_t j = 0; j < xdim.cols; j++)
         if (fabs (_target.at(i, j) - _test.at(i, j)) > _epsilon)
            return false;

   return true;
}

bool
TestLayer::datum_near (const flexnnet::Datum &_target, const flexnnet::Datum &_test, double _epsilon)
{
   std::set<std::string> target_fields = _target.key_set ();
   std::set<std::string> test_fields = _test.key_set();

   // Field names should match
   if (target_fields != test_fields)
      return false;

   // Hash value for fields information should match
   if (_target.hashval() != _test.hashval())
      return false;

   // Field values should match
   for (auto key : _target.key_set ())
   {
      const std::valarray<double>& tgt_valarr = _target[key];
      const std::valarray<double>& tst_valarr = _test[key];

      if (!valarray_double_near(tgt_valarr, tst_valarr, _epsilon))
         return false;
   }

   return true;
}

std::string TestLayer::printResults (const flexnnet::BasicLayer &_layer, int _prec)
{
   std::stringstream ssout;
   ssout.setf (std::ios::fixed, std::ios::floatfield);
   ssout.precision (_prec);

   ssout << prettyPrintVector("output", _layer(), _prec) << "\n";
   ssout << printArray("dAdN", _layer.get_dAdN(), _prec) << "\n";
   ssout << printArray("dNdW", _layer.get_dNdW(), _prec) << "\n";
   ssout << printArray("dNdI", _layer.get_dNdI(), _prec) << "\n";

   cout << ssout.str ();
   return ssout.str();
}


std::string TestLayer::prettyPrintVector(const std::string& _label, const valarray<double>& _vec, int _prec)
{
   std::stringstream ssout;
   ssout .precision (_prec);

   bool first = true;
   ssout << "\n\"" << _label << "\" : \n";
   ssout << "   [";
   for (auto &val : _vec)
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


std::string TestLayer::prettyPrintArray(const std::string& _label, const flexnnet::Array2D<double>& _arr, int _prec)
{
   std::stringstream ssout;
   ssout .precision (_prec);

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

std::string TestLayer::printArray(const std::string& _label, const flexnnet::Array2D<double>& _arr, int _prec)
{
   std::stringstream ssout;
   ssout .precision (_prec);

   Array2D<double>::Dimensions wdim = _arr.size();

   bool first_col = true;
   bool first_row = true;
   ssout << "\n\"" << _label << "\" : \n";
   ssout << "   [";
   for (unsigned int i = 0; i < wdim.rows; i++)
   {
      if (!first_row)
         ssout << "],";
      else
         first_row = false;

      ssout << "[";

      for (unsigned int j = 0; j < wdim.cols; j++)
      {
         if (!first_col)
            ssout << ",";
         else
            first_col = false;
         ssout << _arr.at(i, j);
      }

      first_col = true;
   }
   // end row
   ssout << "]";

   // end array
   ssout << "]";

   return ssout.str();
}