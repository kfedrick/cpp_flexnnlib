//
// Created by kfedrick on 2/17/21.
//

#ifndef _COMMONTESTFIXTURE_H_
#define _COMMONTESTFIXTURE_H_

#include <valarray>
#include <rapidjson/document.h>
#include "flexnnet.h"

#include "Array2D.h"
#include "BasicLayer.h"

class CommonTestFixtureFunctions
{

public:
   static bool
   vector_double_near(const std::valarray<double>& _target, const std::valarray<double>& _test, double _epsilon);

   static bool
   array_double_near(const flexnnet::Array2D<double>& _target, const flexnnet::Array2D<double>& _test, double _epsilon);

   static bool
   valarray_double_near(const std::valarray<double>& _target, const std::valarray<double>& _test, double _epsilon);

   static std::string prettyPrintVector(const std::string& _label, const std::valarray<double>& _vec, int _prec = 4);
   std::string prettyPrintArray(const std::string& _label, const flexnnet::Array2D<double>& _vec, int _prec = 4);
   std::string printResults(const flexnnet::BasicLayer& _layer, int _prec = 5);

   void gday() { std::cout << "g'day\n"; }
   flexnnet::ValarrMap parse_datum(const rapidjson::Value& _obj);
   flexnnet::Array2D<double> parse_weights(const rapidjson::Value& _obj, size_t _rows, size_t _cols);
};

#endif //_COMMONTESTFIXTURE_H_
