//
// Created by kfedrick on 6/18/19.
//

#ifndef _TESTLAYER_H_
#define _TESTLAYER_H_

#include <gtest/gtest.h>
#include <valarray>

#include "Array2D.h"
#include "BasicLayer.h"
#include "OldDatum.h"

class TestLayer
{

public:

   static bool
   datum_near(const flexnnet::OldDatum& _target, const flexnnet::OldDatum& _test, double _epsilon);

   static bool
   vector_double_near(const std::valarray<double>& _target, const std::valarray<double>& _test, double _epsilon);

   static bool
   array_double_near(const flexnnet::Array2D<double>& _target, const flexnnet::Array2D<double>& _test, double _epsilon);

   static bool
   valarray_double_near(const std::valarray<double>& _target, const std::valarray<double>& _test, double _epsilon);

   std::string printArray(const std::string& _label, const flexnnet::Array2D<double>& _vec, int _prec = 4);
   std::string prettyPrintVector(const std::string& _label, const std::valarray<double>& _vec, int _prec = 4);
   std::string prettyPrintArray(const std::string& _label, const flexnnet::Array2D<double>& _vec, int _prec = 4);

protected:
   std::string printResults(const flexnnet::BasicLayer& _layer, int _prec = 5);

};

#endif //_TESTLAYER_H_
