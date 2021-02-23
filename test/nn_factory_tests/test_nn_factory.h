//
// Created by kfedrick on 5/28/19.
//

#ifndef _TEST_NN_FACTORY_H_
#define _TEST_NN_FACTORY_H_

#include <gtest/gtest.h>

#include "BasicNeuralNetFactory.h"

#include "OldLayerConnRecord.h"
#include "BasicLayer.h"

#include "PureLin.h"
#include "LogSig.h"
#include "TanSig.h"
#include "RadBas.h"
#include "SoftMax.h"

#include "deprecated/Pattern.h"
#include "Datum.h"

#include "BasicNeuralNet.h"
#include "BasicNeuralNetSerializer.h"

#include "Array2D.h"

class TestNNFactory
{
};

using std::cout;
using std::endl;
using std::string;
using flexnnet::Datum;
using flexnnet::Array2D;

using flexnnet::OldLayerConnRecord;
using flexnnet::BasicLayer;;
using flexnnet::BasicNeuralNetFactory;
using flexnnet::BasicNeuralNetSerializer;

class TestBasicNNFactory : public TestNNFactory, public ::testing::TestWithParam<const char*>
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   void printArray2D(const std::string& _label, Array2D<double>& _arr);

   const size_t SINGLE_SZ = 3;
   const BasicLayer::NetworkLayerType SINGLE_LTYPE = BasicLayer::Output;
   const bool SINGLE_IS_OUTPUT = (SINGLE_LTYPE == BasicLayer::Output);

   const string SINGLE_PURELIN_ID = "purelin_layer";
   const string SINGLE_TANSIG_ID = "tagsig_layer";
   const string SINGLE_RADBAS_ID = "radbas_layer";
   const string SINGLE_SOFTMAX_ID = "softmax_layer";
   const string SINGLE_LOGSIG_ID = "logsig_layer";
};

inline void TestBasicNNFactory::printArray2D(const std::string& _label, Array2D<double>& _arr)
{
   printf("\n\n");
   printf("%s\n", _label.c_str());
   Array2D<double>::Dimensions dim = _arr.size();
   for (unsigned int i = 0; i < dim.rows; i++)
   {
      for (unsigned int j = 0; j < dim.cols; j++)
         printf("%7.5f ", _arr.at(i, j));
      printf("\n");
   }
   printf("---\n\n");
}

#endif //_TEST_NN_FACTORY_H_
