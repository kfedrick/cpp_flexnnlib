//
// Created by kfedrick on 6/25/19.
//

#ifndef _TEST_NN_ACTIVATION_H_
#define _TEST_NN_ACTIVATION_H_

#include <valarray>

class TestNNActivation : public ::testing::TestWithParam<const char*>
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}


   std::string prettyPrintVector(const std::string& _label, const std::valarray<double>& _vec, int _prec)
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
   };
};

#endif //_TEST_NN_ACTIVATION_H_
