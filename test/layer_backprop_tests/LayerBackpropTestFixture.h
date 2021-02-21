//
// Created by kfedrick on 2/17/21.
//

#ifndef _LAYERBACKPROPTESTFIXTURE_H_
#define _LAYERBACKPROPTESTFIXTURE_H_

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/istreamwrapper.h>
#include <CommonTestFixtureFunctions.h>

#include "PureLin.h"
#include "LogSig.h"

using flexnnet::PureLin;
using flexnnet::LogSig;

template<class _LayerType>
class LayerBackpropTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

public:
   std::shared_ptr<_LayerType> layer_ptr;

   struct TestCaseRecord
   {
      std::valarray<double> initial_value;
      std::valarray<double> input;

      struct Target
      {
         std::valarray<double> output;
         flexnnet::Array2D<double> dAdN;
         flexnnet::Array2D<double> dNdW;
         flexnnet::Array2D<double> dNdI;
      };

      Target target;
   };
};

TYPED_TEST_CASE_P(LayerBackpropTestFixture);

TYPED_TEST_P(LayerBackpropTestFixture, FooTest)
{
   typename LayerBackpropTestFixture<TypeParam>::TestCaseRecord rec;
   rec.input = {1, 2, 3};
   rec.target.dAdN.resize(2,2);
   rec.target.dAdN = {{1, 2}, {3, 4}};
   rec.target.dAdN.set({{1, 2, 3}, {3, 4, 5}});

   flexnnet::LayerWeights y({{1,2}, {3,4}});

   rec.target.dAdN.size();

   std::vector<std::vector<double>> x = {{1, 2}, {3, 4}};
   flexnnet::Array2D<double>::Dimensions d = flexnnet::Array2D<double>::size(x);
   std::cout << "(" << d.rows << ", " << d.cols << ")\n";

   d = rec.target.dAdN.size();
   std::cout << "(" << d.rows << ", " << d.cols << ")\n";

   this->layer_ptr = std::make_shared<TypeParam>(3, "purelin");
   std::cout << this->layer_ptr->name().c_str() << "\n";
   ASSERT_TRUE(true);
}

/* Registration and instantiation of type-paramed tests */
REGISTER_TYPED_TEST_CASE_P(LayerBackpropTestFixture, FooTest);

typedef ::testing::Types<PureLin> MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Sanity, LayerBackpropTestFixture, MyTypes );

#endif //_LAYERBACKPROPTESTFIXTURE_H_
