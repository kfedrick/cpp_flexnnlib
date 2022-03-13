//
// Created by kfedrick on 5/25/21.
//

#ifndef FLEX_NEURALNET_TESTACTIONFEATURE_H_
#define FLEX_NEURALNET_TESTACTIONFEATURE_H_

#include <Feature.h>
#include <RawFeature.h>

class TestActionFeature : public flexnnet::RawFeature<1>
{
public:
   TestActionFeature(flexnnet::RawFeature<1> _feature);

   enum class ActionEnum { Left, Right };

public:
   TestActionFeature();
   virtual const TestActionFeature::ActionEnum& get_action() const;
   virtual void decode(const std::valarray<double>& _encoding);

   TestActionFeature& operator=(const TestActionFeature& _f);

private:
   ActionEnum action;
};

TestActionFeature::TestActionFeature() : RawFeature<1>()
{
   action = ActionEnum::Right;
}

TestActionFeature::TestActionFeature(flexnnet::RawFeature<1> _feature) : RawFeature<1>()
{
   flexnnet::RawFeature<1>::copy(_feature);
}

TestActionFeature& TestActionFeature::operator=(const TestActionFeature& _f)
{
   flexnnet::RawFeature<1>::copy(_f);
   return *this;
}

const TestActionFeature::ActionEnum& TestActionFeature::get_action() const
{
   return action;
}

void TestActionFeature::decode(const std::valarray<double>& _encoding)
{
   Feature::decode(_encoding);

   if (const_encoding_ref[0] > 0)
      action = ActionEnum::Right;
   else
      action = ActionEnum::Left;
}

#endif // FLEX_NEURALNET_TESTACTIONFEATURE_H_
