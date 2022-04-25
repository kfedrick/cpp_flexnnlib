//
// Created by kfedrick on 5/25/21.
//

#ifndef FLEX_NEURALNET_STEERINGACTIONFEATURE_H_
#define FLEX_NEURALNET_STEERINGACTIONFEATURE_H_

#include <Feature.h>
#include <RawFeature.h>
#include <random>

class SteeringActionFeature : public flexnnet::RawFeature<1>
{
public:
   SteeringActionFeature(flexnnet::RawFeature<1> _feature);

   enum class ActionEnum { Left, Right };
   struct ActionDetails {};

public:
   SteeringActionFeature();
   virtual const ActionEnum& get_action() const;
   virtual const ActionDetails& get_action_details() const;

   virtual void decode(const std::valarray<double>& _encoding);

   SteeringActionFeature& operator=(const SteeringActionFeature& _f);

private:
   mutable ActionEnum action;
   ActionDetails details;
   mutable std::mt19937_64 rand_engine;

};

inline
SteeringActionFeature::SteeringActionFeature() : RawFeature<1>()
{
   action = ActionEnum::Right;

   std::random_device r;
   std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
   rand_engine.seed(seed2);


}

inline
SteeringActionFeature::SteeringActionFeature(flexnnet::RawFeature<1> _feature) : RawFeature<1>()
{
   flexnnet::RawFeature<1>::copy(_feature);
}

inline
SteeringActionFeature& SteeringActionFeature::operator=(const SteeringActionFeature& _f)
{
   flexnnet::RawFeature<1>::copy(_f);
   action = _f.action;
   return *this;
}

inline
const SteeringActionFeature::ActionEnum& SteeringActionFeature::get_action() const
{
   std::normal_distribution<double> normal_dist(0, 0.01);
   normal_dist(rand_engine);

   if (const_encoding_ref[0] +  normal_dist(rand_engine) > 0)
      action = ActionEnum::Right;
   else
      action = ActionEnum::Left;

   return action;
}

inline
void SteeringActionFeature::decode(const std::valarray<double>& _encoding)
{
   double r = 2 * ((double) rand()) / RAND_MAX - 0.5;
   //Feature::decode({r});

   Feature::decode(_encoding);
   //std::cout << "decoding action feature " << const_encoding_ref[0] << "\n" << std::flush;

   if (const_encoding_ref[0] > 0)
      action = ActionEnum::Right;
   else
      action = ActionEnum::Left;
}

inline
const SteeringActionFeature::ActionDetails& SteeringActionFeature::get_action_details() const
{
   return details;
}

#endif // FLEX_NEURALNET_TESTACTIONFEATURE_H_
