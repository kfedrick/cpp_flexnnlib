//
// Created by kfedrick on 6/19/19.
//

#include "test_nn_serialization.h"
#include "TestLayer.h"

#include "PureLin.h"
#include "LogSig.h"
#include "SoftMax.h"

#include "VectorizableSet.h"
#include "A.h"

using std::cout;

using flexnnet::Datum;
using flexnnet::BasicLayer;
using flexnnet::SoftMax;
using flexnnet::BasicNeuralNetFactory;
using flexnnet::BasicNeuralNetSerializer;
using flexnnet::NeuralNetSerializer;

static bool
array_double_near(const flexnnet::Array2D<double>& _target, const flexnnet::Array2D<double>& _test, double _epsilon);

TEST_F(TestBasicNeuralNet, Serializer)
{
   std::string target_json = "{\"neuralnet_id\":\"net\",\"network_layers\":[{\"id\":\"hidden1\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::SoftMax\",\"parameters\":{\"gain\":1.631,\"rescaled\":true}}},{\"id\":\"hidden2\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::PureLin\",\"parameters\":{\"gain\":1.0}}},{\"id\":\"hidden3\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::PureLin\",\"parameters\":{\"gain\":1.0}}},{\"id\":\"output\",\"is_output_layer\":true,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::LogSig\",\"parameters\":{\"gain\":1.0}}}],\"layer_topology\":[{\"id\":\"hidden1\",\"input_connections\":[],\"external_inputs\":[{\"field\":\"i1\",\"index\":0,\"size\":3}]},{\"id\":\"hidden2\",\"input_connections\":[{\"id\":\"hidden1\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]},{\"id\":\"hidden3\",\"input_connections\":[{\"id\":\"hidden2\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]},{\"id\":\"output\",\"input_connections\":[{\"id\":\"hidden3\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]}],\"network_input\":[{\"field\":\"i3\",\"index\":2,\"size\":2},{\"field\":\"i2\",\"index\":1,\"size\":5},{\"field\":\"i1\",\"index\":0,\"size\":3}]}";

   BasicNeuralNetFactory factory;

   std::valarray<double> v1(3);
   std::valarray<double> v2(5);
   std::valarray<double> v3(2);

   Datum inpatt({{"i1", v1}, {"i2", v2}, {"i3", v3}});

   std::valarray<double> v4(10);

   Datum inpatt2({{"i1", v1}, {"i2", v2}, {"i3", v3}});


//   factory.set_network_input(inpatt);

   flexnnet::SoftMax::Parameters p = {.gain=1.631, .rescaled_flag=true};

   factory.add_layer<flexnnet::SoftMax>(3, "hidden1", BasicLayer::Hidden, p);
   factory.add_layer<flexnnet::PureLin>(3, "hidden2", BasicLayer::Hidden);
   factory.add_layer<flexnnet::PureLin>(3, "hidden3", BasicLayer::Hidden);
   factory.add_layer<flexnnet::LogSig>(3, "output", BasicLayer::Output);

   factory.set_layer_external_input("hidden1", inpatt2, std::set<std::string>({"i1"}));

   factory.add_layer_connection("hidden2", "hidden1");
   factory.add_layer_connection("hidden3", "hidden2");
   factory.add_layer_connection("output", "hidden3");

   std::shared_ptr<flexnnet::BasicNeuralNet> net = factory.build("net");

   cout << "Print network layer names\n";
   std::set<std::string> names = net->get_layer_names();
   for (auto& layer_name : names)
      cout << layer_name << "\n";

   //std::string json = net->toJSON();
   //cout << json << "\n";

   //ASSERT_EQ(target_json, json);
}

TEST_F(TestBasicNeuralNet, Deserializer)
{
   //flexnnet::OldVectorSet<int, float, char, double, bool> vs(666, 3.14159, 'a', 2.7, true);
   //flexnnet::OldVectorSet<int> vs(666);

   flexnnet::A a("bugs1", {0, 3.14159, 2});
   flexnnet::A b("bugs2", {666, 2.7});

   std::cout << "call OldVectorSet constructor\n";
   //flexnnet::VectorizableSet<flexnnet::A, flexnnet::A> vs(a, flexnnet::A("bugs", {666, 2.7}));
   flexnnet::VectorizableSet<flexnnet::A, flexnnet::A> vs(a, b);

   //flexnnet::OldVectorSet<flexnnet::A> vs(flexnnet::A("a_vectorizable", {0, 1.3, 2}));

   std::cout << "vs[a_vec][1] " << vs.at("bugs1")[1] << "\n";
   std::cout << "vs[bugs][1] " << vs.at("bugs2")[1] << "\n";

   //std::cout << "print vectorset values\n";
   //vs.doit();
   //std::cout << "\n\n";

   std::string target_json = "{\"neuralnet_id\":\"net\",\"network_layers\":[{\"id\":\"hidden1\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::SoftMax\",\"parameters\":{\"gain\":1.631,\"rescaled\":true}}},{\"id\":\"hidden2\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::PureLin\",\"parameters\":{\"gain\":1.0}}},{\"id\":\"hidden3\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::PureLin\",\"parameters\":{\"gain\":1.0}}},{\"id\":\"output\",\"is_output_layer\":true,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::LogSig\",\"parameters\":{\"gain\":1.0}}}],\"layer_topology\":[{\"id\":\"hidden1\",\"input_connections\":[],\"external_inputs\":[{\"field\":\"i1\",\"index\":0,\"size\":3}]},{\"id\":\"hidden2\",\"input_connections\":[{\"id\":\"hidden1\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]},{\"id\":\"hidden3\",\"input_connections\":[{\"id\":\"hidden2\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]},{\"id\":\"output\",\"input_connections\":[{\"id\":\"hidden3\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]}],\"network_input\":[{\"field\":\"i3\",\"index\":2,\"size\":2},{\"field\":\"i2\",\"index\":1,\"size\":5},{\"field\":\"i1\",\"index\":0,\"size\":3}]}";

   cout << "deserialize\n";
   std::shared_ptr<flexnnet::NeuralNet<Datum, Datum>> net = NeuralNetSerializer<Datum, Datum>::parse(target_json);

   cout << "Print network layer names\n";
   std::set<std::string> names = net->get_layer_names();
   for (auto& layer_name : names)
      cout << layer_name << "\n";

   cout << "stuff\n";
}

TEST_F(TestBasicNeuralNet, SaveNetworkWeights)
{
   cout << "SaveNetworkWeights Test\n";

   std::string target_json = "{\"neuralnet_id\":\"net\",\"network_layers\":[{\"id\":\"hidden1\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.5,0.0,0.2,0.0],[0.0,8.0,0.0,0.0],[0.7,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::SoftMax\",\"parameters\":{\"gain\":1.631,\"rescaled\":true}}},{\"id\":\"hidden2\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::PureLin\",\"parameters\":{\"gain\":1.0}}},{\"id\":\"hidden3\",\"is_output_layer\":false,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[1.0,0.0,0.0,0.0],[0.0,3.14,0.0,0.0],[0.0,0.0,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::PureLin\",\"parameters\":{\"gain\":1.0}}},{\"id\":\"output\",\"is_output_layer\":true,\"dimensions\":{\"layer_size\":3,\"layer_input_size\":3},\"learned_parameters\":{\"weights\":[[0.0,6.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.127,0.0,0.0]]},\"transfer_function\":{\"type\":\"flexnnet::LogSig\",\"parameters\":{\"gain\":1.0}}}],\"layer_topology\":[{\"id\":\"hidden1\",\"input_connections\":[],\"external_inputs\":[{\"field\":\"i1\",\"index\":0,\"size\":3}]},{\"id\":\"hidden2\",\"input_connections\":[{\"id\":\"hidden1\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]},{\"id\":\"hidden3\",\"input_connections\":[{\"id\":\"hidden2\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]},{\"id\":\"output\",\"input_connections\":[{\"id\":\"hidden3\",\"conn_type\":\"forward\",\"size\":3}],\"external_inputs\":[]}],\"network_input\":[{\"field\":\"i3\",\"index\":2,\"size\":2},{\"field\":\"i2\",\"index\":1,\"size\":5},{\"field\":\"i1\",\"index\":0,\"size\":3}]}";

   cout << "deserialize\n";
   std::shared_ptr<flexnnet::NeuralNet<Datum, Datum>> net = NeuralNetSerializer<Datum, Datum>::parse(target_json);

   cout << "Print network layer names\n";
   std::set<std::string> names = net->get_layer_names();
   for (auto& layer_name : names)
      cout << layer_name << "\n";

   flexnnet::Array2D<double> myweights(3, 4);
   myweights.at(0, 0) = 1;
   myweights.at(1, 1) = 2;
   myweights.at(2, 2) = 3;

   net->get_layers()[0]->layer_weights.set_weights(myweights);
   net->get_layers()[1]->layer_weights.set_weights(myweights);
   net->get_layers()[2]->layer_weights.set_weights(myweights);
   net->get_layers()[3]->layer_weights.set_weights(myweights);

   NetworkWeights network_weights = net->get_weights();

   const flexnnet::LayerWeights& lweights = network_weights["hidden2"];

   for (size_t row = 0; row < lweights.const_weights_ref.size().rows; row++)
   {
      for (size_t col = 0; col < lweights.const_weights_ref.size().cols; col++)
         cout << lweights.const_weights_ref(row, col) << " ";
      cout << "\n";
   }
   cout << "\n";

   for (auto alayer_ptr : net->get_layers())
      alayer_ptr->layer_weights.zero();

   net->set_weights(network_weights);

   for (auto alayer_ptr : net->get_layers())
   {
      const flexnnet::Array2D<double>& item = alayer_ptr->layer_weights.const_weights_ref;
      const flexnnet::Array2D<double>& target = network_weights[alayer_ptr->name()].const_weights_ref;
      EXPECT_PRED3(array_double_near, item, target, 0.000000001) << "ruh roh";
   }

}

int main(int argc, char** argv)
{
   ::testing::InitGoogleTest(&argc, argv);

   return RUN_ALL_TESTS();
}