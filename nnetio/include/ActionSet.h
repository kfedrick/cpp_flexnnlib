//
// Created by kfedrick on 5/31/21.
//

#ifndef FLEX_NEURALNET_ACTIONSET_H_
#define FLEX_NEURALNET_ACTIONSET_H_

#include <FeatureSetImpl.h>

template<class ActionFeature>
class ActionSet : public flexnnet::FeatureSetImpl<std::tuple<ActionFeature>>
{
public:
   typedef typename ActionFeature::ActionEnum ActionEnum;

public:
   ActionSet();
   ActionSet(const flexnnet::FeatureSetImpl<std::tuple<ActionFeature>>& _actionset);
   ActionSet(const ActionSet<ActionFeature>& _actionset);
   ActionSet& operator=(const ActionSet<ActionFeature>& _actionset);
   ActionSet& operator=(const flexnnet::FeatureSetImpl<std::tuple<ActionFeature>>& _actionset);

   const typename ActionFeature::ActionEnum get_action() const;
   const typename ActionFeature::ActionDetails get_action_details() const;
};

template<class F>
ActionSet<F>::ActionSet() : flexnnet::FeatureSetImpl<std::tuple<F>>()
{
}

template<class F>
ActionSet<F>::ActionSet(const flexnnet::FeatureSetImpl<std::tuple<F>>& _fs) : flexnnet::FeatureSetImpl<std::tuple<F>>(_fs)
{
}

template<class F>
ActionSet<F>::ActionSet(const ActionSet<F>& _as) : flexnnet::FeatureSetImpl<std::tuple<F>>(_as)
{
}

template<class F>
ActionSet<F>& ActionSet<F>::operator=(const ActionSet<F>& _as)
{
   flexnnet::FeatureSetImpl<std::tuple<F>>::operator=((const flexnnet::FeatureSetImpl<std::tuple<F>>&) _as);
   return *this;
}

template<class F>
ActionSet<F>& ActionSet<F>::operator=(const flexnnet::FeatureSetImpl<std::tuple<F>>& _as)
{
   flexnnet::FeatureSetImpl<std::tuple<F>>::operator=(_as);
   return *this;
}

template<class F>
const typename F::ActionEnum ActionSet<F>::get_action() const
{
   return std::get<0>(this->get_features()).get_action();
}

#endif // FLEX_NEURALNET_ACTIONSET_H_
