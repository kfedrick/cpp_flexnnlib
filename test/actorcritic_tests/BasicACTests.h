//
// Created by kfedrick on 5/17/21.
//

#ifndef _BASICACTESTS_H_
#define _BASICACTESTS_H_

#include <gtest/gtest.h>
#include <CommonTestFixtureFunctions.h>

#include <Thing.h>
#include <Thang.h>
#include <FeatureVector.h>
#include <BaseActorCriticNetwork.h>
#include <NetworkLayerImpl.h>
#include <BaseNeuralNet.h>
#include <NeuralNet.h>
#include <VariadicNetworkInput.h>

#include <TanSig.h>
#include <NetworkReinforcement.h>

class BasicACTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   virtual void
   SetUp()
   {}
   virtual void
   TearDown()
   {}
};

TEST_F(BasicACTestFixture, UnpackTest)
{
   enum class ActionEnum { Left, Right };
   flexnnet::Thing thing1("one");
   flexnnet::Thang thang2("two");
   flexnnet::Thing thing3("three");

   flexnnet::BaseActorCriticNetwork<ActionEnum, flexnnet::FeatureVector, flexnnet::Thing, flexnnet::Thang, flexnnet::Thing> nn;

   nn.activate(thing1, thang2, thing3);
}

class AFunctor
{
public:
   template<typename T>
   void operator()(const T& _t)
   {
      std::cout << "this is it " << _t << "\n";
   }
};

template<typename ...Fs>
class FeatureVec
{
public:
   int count() const
   {
      return sizeof...(Fs);
   }

   void printit() const
   {
      print(data);
   }

   void foreach() const
   {
      AFunctor afunctor;
      for_each(data, afunctor);
   }

protected:


   //template<std::size_t I = 0, typename... Tp>
   template<std::size_t I = 0>
   typename std::enable_if<I == sizeof...(Fs), void>::type
   print(const std::tuple<Fs...>& t) const
   { }

   template<std::size_t I = 0>
   typename std::enable_if<I < sizeof...(Fs), void>::type
   print(const std::tuple<Fs...>& t) const
   {
      std::cout << std::get<I>(t) << std::endl;
      print<I + 1>(t);
   }

   // ******************************************
   template<std::size_t I = 0, typename FuncT>
   inline typename std::enable_if<I == sizeof...(Fs), void>::type
   for_each(const std::tuple<Fs...> &, FuncT) const // Unused arguments are given no names.
   { }

   template<std::size_t I = 0, typename FuncT>
   inline typename std::enable_if<I < sizeof...(Fs), void>::type
   for_each(const std::tuple<Fs...>& t, FuncT f) const
   {
      f(std::get<I>(t));
      for_each<I + 1, FuncT>(t, f);
   }

public:
   std::tuple<Fs...> data;
};



template<typename Input>
class MockNeuralNet
{
public:
   MockNeuralNet(const Input& _in)
   {
      std::cout << _in.count() << "\n";
      _in.printit();
      _in.foreach();
   }
};

template<size_t N>
class BasicFeature : public flexnnet::Vectorizable
{
public:
   BasicFeature() : Vectorizable()
   {}

   const std::valarray<double>& vectorize() const
   {
      return data;
   }

   const Vectorizable& assign(const std::valarray<double>& _val)
   {
      return *this;
   };

   Vectorizable operator=(const BasicFeature& _f)
   {
      data = _f.data;
      return *this;
   }

   std::valarray<double> data;
};

TEST_F(BasicACTestFixture, Constructor)
{
   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol_ptr = std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "output", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 1);

   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::BaseNeuralNet basennet(topo);

   FeatureVec<int,double,char> a_featurevec;
   std::get<0>(a_featurevec.data) = 666;
   MockNeuralNet<FeatureVec<int,double,char>> mocknnet(a_featurevec);
}

TEST_F(BasicACTestFixture, VariadicNullConstructor)
{
   std::cout << "VariadicNullConstructor Test\n" << std::flush;

   BasicFeature<3> f1;
   BasicFeature<5> f2;
   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v;
}

TEST_F(BasicACTestFixture, VariadicConstructor)
{
   std::cout << "VariadicPairConstructor Test\n" << std::flush;

   BasicFeature<3> f1; f1.data = {1,2,3};
   BasicFeature<5> f2; f2.data = {0.1, -1.3, 3.14159, 2.17, 0};
   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v(std::pair<std::string,BasicFeature<3>>("first", f1), std::pair<std::string,BasicFeature<5>>("second", f2));

   flexnnet::ValarrMap vmap = v.value_map();
   std::cout << this->prettyPrintVector("first", vmap["first"]) << "\n";
   std::cout << this->prettyPrintVector("second", vmap["second"]) << "\n";
}

TEST_F(BasicACTestFixture, VariadicAsNetworkInputRef)
{
   std::cout << "VariadicAsNetworkInputRef Test\n" << std::flush;

   BasicFeature<3> f1; f1.data = {1,2,3};
   BasicFeature<5> f2; f2.data = {0.1, -1.3, 3.14159, 2.17, 0};
   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v(std::pair<std::string,BasicFeature<3>>("first", f1), std::pair<std::string,BasicFeature<5>>("second", f2));

   flexnnet::NetworkInput& ni = v;

   flexnnet::ValarrMap vmap = ni.value_map();
   std::cout << this->prettyPrintVector("first", vmap["first"]) << "\n";
   std::cout << this->prettyPrintVector("second", vmap["second"]) << "\n";
}

TEST_F(BasicACTestFixture, VariadicAsNetworkInputPtr)
{
   std::cout << "VariadicAsNetworkInputPtr Test\n" << std::flush;

   BasicFeature<3> f1; f1.data = {1,2,3};
   BasicFeature<5> f2; f2.data = {0.1, -1.3, 3.14159, 2.17, 0};
   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v(std::pair<std::string,BasicFeature<3>>("first", f1), std::pair<std::string,BasicFeature<5>>("second", f2));

   flexnnet::NetworkInput* ni = &v;

   flexnnet::ValarrMap vmap = ni->value_map();
   std::cout << this->prettyPrintVector("first", vmap["first"]) << "\n";
   std::cout << this->prettyPrintVector("second", vmap["second"]) << "\n";
}

TEST_F(BasicACTestFixture, VariadicCopyConstructor)
{
   std::cout << "VariadicCopyConstructor Test\n" << std::flush;

   BasicFeature<3> f1; f1.data = {1,2,3};
   BasicFeature<5> f2; f2.data = {0.1, -1.3, 3.14159, 2.17, 0};
   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v(std::pair<std::string,BasicFeature<3>>("first", f1), std::pair<std::string,BasicFeature<5>>("second", f2));

   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v2(v);

   flexnnet::ValarrMap vmap2 = v2.value_map();
   std::cout << this->prettyPrintVector("first", vmap2["first"]) << "\n";
   std::cout << this->prettyPrintVector("second", vmap2["second"]) << "\n";
}

TEST_F(BasicACTestFixture, VariadicAssignment)
{
   std::cout << "VariadicAssign Test\n" << std::flush;

   BasicFeature<3> f1; f1.data = {1,2,3};
   BasicFeature<5> f2; f2.data = {0.1, -1.3, 3.14159, 2.17, 0};
   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v(std::pair<std::string,BasicFeature<3>>("first", f1), std::pair<std::string,BasicFeature<5>>("second", f2));

   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v2;
   v2 = v;

   flexnnet::ValarrMap vmap2 = v2.value_map();
   std::cout << this->prettyPrintVector("first", vmap2["first"]) << "\n";
   std::cout << this->prettyPrintVector("second", vmap2["second"]) << "\n";
}

TEST_F(BasicACTestFixture, VariadicSetFromTuple)
{
   std::cout << "VariadicSetFromTuple Test\n" << std::flush;

   BasicFeature<3> f1; f1.data = {0,0,0};
   BasicFeature<5> f2; f2.data = {0,0,0,0,0};
   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v(std::pair<std::string,BasicFeature<3>>("first", f1), std::pair<std::string,BasicFeature<5>>("second", f2));

   f1.data = {1,2,3};
   f2.data = {0.1, -1.3, 3.14159, 2.17, 0};

   std::tuple<BasicFeature<3>,BasicFeature<5>> t(f1,f2);
   v.set(t);

   flexnnet::ValarrMap vmap = v.value_map();
   std::cout << this->prettyPrintVector("first", vmap["first"]) << "\n";
   std::cout << this->prettyPrintVector("second", vmap["second"]) << "\n";
}

TEST_F(BasicACTestFixture, VariadicSetFromPack)
{
   std::cout << "VariadicSetFromPack Test\n" << std::flush;

   BasicFeature<3> f1; f1.data = {0,0,0};
   BasicFeature<5> f2; f2.data = {0,0,0,0,0};
   flexnnet::VariadicNetworkInput<BasicFeature<3>, BasicFeature<5>> v(std::pair<std::string,BasicFeature<3>>("first", f1), std::pair<std::string,BasicFeature<5>>("second", f2));

   f1.data = {1,2,3};
   f2.data = {0.1, -1.3, 3.14159, 2.17, 0};

   v.set(f1,f2);

   flexnnet::ValarrMap vmap = v.value_map();
   std::cout << this->prettyPrintVector("first", vmap["first"]) << "\n";
   std::cout << this->prettyPrintVector("second", vmap["second"]) << "\n";
}

TEST_F(BasicACTestFixture, VariadicModSingleFeature)
{
   std::cout << "VariadicModSingleFeature Test\n" << std::flush;

   std::valarray<double> va = {0,0,0};
   BasicFeature<5> f2; f2.data = {0,0,0,0,0};
   flexnnet::VariadicNetworkInput<std::valarray<double>, BasicFeature<5>> v(std::pair<std::string,std::valarray<double>>("first", va), std::pair<std::string,BasicFeature<5>>("second", f2));

   std::tuple<std::valarray<double>, BasicFeature<5>>& t = v.values();
   std::get<0>(t)[0] = 1;

   flexnnet::ValarrMap vmap = v.value_map();
   std::cout << this->prettyPrintVector("first", vmap["first"]) << "\n";
   std::cout << this->prettyPrintVector("second", vmap["second"]) << "\n";
}

TEST_F(BasicACTestFixture, CriticNetConstructor)
{
   std::cout << "Critic network Constructor Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol_ptr =
                                                std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "output", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 1);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<flexnnet::FeatureVector, flexnnet::NetworkReinforcement<1>> critic(basecritic);
}

TEST_F(BasicACTestFixture, SingleCriticActivate)
{
   std::cout << "Single Critic Activate Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "output", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol_ptr->add_external_input_field("input", 3);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol_ptr->name()] = ol_ptr;
   topo.network_output_layers.push_back(ol_ptr);
   topo.ordered_layers.push_back(ol_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<flexnnet::FeatureVector, flexnnet::NetworkReinforcement<1>> critic(basecritic);
   critic.initialize_weights();

   flexnnet::FeatureVector f({{"input",{0, 1, 2}}});
   critic.activate(f);

   flexnnet::NetworkReinforcement<1> nnout = critic.activate(f);
   std::cout << prettyPrintVector("nnout", nnout.value());
}

TEST_F(BasicACTestFixture, MultiCriticActivate)
{
   std::cout << "Multi Critic Activate Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol1_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "reinf1", flexnnet::TanSig::DEFAULT_PARAMS, true));
   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol2_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "reinf2", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol1_ptr->add_external_input_field("input", 3);
   ol2_ptr->add_external_input_field("input", 3);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol1_ptr->name()] = ol1_ptr;
   topo.network_layers[ol2_ptr->name()] = ol2_ptr;

   topo.network_output_layers.push_back(ol1_ptr);
   topo.network_output_layers.push_back(ol2_ptr);

   topo.ordered_layers.push_back(ol1_ptr);
   topo.ordered_layers.push_back(ol2_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<flexnnet::FeatureVector, flexnnet::NetworkReinforcement<2>> critic(basecritic);
   critic.initialize_weights();

   flexnnet::FeatureVector f({{"input",{0, 1, 2}}});
   critic.activate(f);

   flexnnet::NetworkReinforcement<2> nnout = critic.activate(f);
   std::cout << prettyPrintVector("nnout", nnout.value());
}

TEST_F(BasicACTestFixture, MultiCriticAccessField)
{
   std::cout << "Multi Critic Access by field Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol1_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "reinf1", flexnnet::TanSig::DEFAULT_PARAMS, true));
   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol2_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "reinf2", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol1_ptr->add_external_input_field("input", 3);
   ol2_ptr->add_external_input_field("input", 3);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol1_ptr->name()] = ol1_ptr;
   topo.network_layers[ol2_ptr->name()] = ol2_ptr;

   topo.network_output_layers.push_back(ol1_ptr);
   topo.network_output_layers.push_back(ol2_ptr);

   topo.ordered_layers.push_back(ol1_ptr);
   topo.ordered_layers.push_back(ol2_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<flexnnet::FeatureVector, flexnnet::NetworkReinforcement<2>> critic(basecritic);
   critic.initialize_weights();

   flexnnet::FeatureVector f({{"input",{0, 1, 2}}});
   critic.activate(f);

   flexnnet::NetworkReinforcement<2> nnout = critic.activate(f);

   std::vector<std::string> fields = nnout.get_fields();
   for (auto a_field : fields)
   {
      std::cout << a_field << " " << nnout.at(a_field) << "\n";
   }
}

TEST_F(BasicACTestFixture, MultiCriticAccessIndex)
{
   std::cout << "Multi Critic Access by Index Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol1_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "reinf1", flexnnet::TanSig::DEFAULT_PARAMS, true));
   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol2_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "reinf2", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol1_ptr->add_external_input_field("input", 3);
   ol2_ptr->add_external_input_field("input", 3);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol1_ptr->name()] = ol1_ptr;
   topo.network_layers[ol2_ptr->name()] = ol2_ptr;

   topo.network_output_layers.push_back(ol1_ptr);
   topo.network_output_layers.push_back(ol2_ptr);

   topo.ordered_layers.push_back(ol1_ptr);
   topo.ordered_layers.push_back(ol2_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<flexnnet::FeatureVector, flexnnet::NetworkReinforcement<2>> critic(basecritic);
   critic.initialize_weights();

   flexnnet::FeatureVector f({{"input",{0, 1, 2}}});
   critic.activate(f);

   flexnnet::NetworkReinforcement<2> nnout = critic.activate(f);

   std::vector<std::string> fields = nnout.get_fields();
   for (int ndx=0; ndx<fields.size(); ndx++)
   {
      std::cout << ndx << " " << fields[ndx] << " " << nnout[ndx] << "\n";
   }
}

TEST_F(BasicACTestFixture, MultiCriticAccessIndexAt)
{
   std::cout << "Multi Critic Access by Index at Test\n" << std::flush;

   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol1_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "reinf1", flexnnet::TanSig::DEFAULT_PARAMS, true));
   std::shared_ptr<flexnnet::NetworkLayerImpl<flexnnet::TanSig>> ol2_ptr =
      std::make_shared<flexnnet::NetworkLayerImpl<flexnnet::TanSig>>(flexnnet::NetworkLayerImpl<flexnnet::TanSig>(1, "reinf2", flexnnet::TanSig::DEFAULT_PARAMS, true));
   ol1_ptr->add_external_input_field("input", 3);
   ol2_ptr->add_external_input_field("input", 3);


   flexnnet::NeuralNetTopology topo;
   topo.network_layers[ol1_ptr->name()] = ol1_ptr;
   topo.network_layers[ol2_ptr->name()] = ol2_ptr;

   topo.network_output_layers.push_back(ol1_ptr);
   topo.network_output_layers.push_back(ol2_ptr);

   topo.ordered_layers.push_back(ol1_ptr);
   topo.ordered_layers.push_back(ol2_ptr);

   flexnnet::BaseNeuralNet basecritic(topo);
   flexnnet::NeuralNet<flexnnet::FeatureVector, flexnnet::NetworkReinforcement<2>> critic(basecritic);
   critic.initialize_weights();

   flexnnet::FeatureVector f({{"input",{0, 1, 2}}});
   critic.activate(f);

   flexnnet::NetworkReinforcement<2> nnout = critic.activate(f);

   EXPECT_EQ(nnout.size(), 2) << "Reinforcement size not correct\n";

   std::vector<std::string> fields = nnout.get_fields();
   for (int ndx=0; ndx<fields.size(); ndx++)
   {
      std::cout << ndx << " " << fields[ndx] << " " << nnout.at(ndx) << "\n";
   }
}
#endif //_BASICACTESTS_H_
