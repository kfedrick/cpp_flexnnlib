//
// Created by kfedrick on 2/20/21.
//

#ifndef _NNBUILDERTESTFIXTURE_H_
#define _NNBUILDERTESTFIXTURE_H_

#include <CommonTestFixtureFunctions.h>

#include "PureLin.h"
#include "LayerConnRecord.h"
#include "NetworkLayer.h"
#include "NetworkTopology.h"

using flexnnet::PureLin;
using flexnnet::LayerConnRecord;
using flexnnet::NetworkLayer;
using flexnnet::NetworkTopology;

class NNBuilderTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:

   NNBuilderTestFixture()
   {
      sample_external_input = {{"a", {1, 2}}, {"b", {0.1, -2, 1.5}}, {"c", {1e10, 2.5, 666, 7}}};
   }

   virtual void SetUp()
   {}
   virtual void TearDown()
   {}

   std::map<std::string, std::vector<double>> sample_external_input;
};

TEST_F(NNBuilderTestFixture, NetworkLayerDefaultCon)
{
   try
   {
      NetworkLayer nl;
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NetworkLayerNetLayerSetLayerPtr)
{
   try
   {
      NetworkLayer nl(std::shared_ptr<flexnnet::BasicLayer>(new PureLin(3, "purelin")));
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NetworkLayerGetActConn)
{
   try
   {
      NetworkLayer nl(std::shared_ptr<flexnnet::BasicLayer>(new PureLin(3, "purelin")));
      const std::vector<LayerConnRecord>& conn = nl.get_activation_connections();

      ASSERT_EQ(conn.size(), 0) << "Expected network connection size to be zero, got " << conn.size() << "\n";
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NetworkLayerGetExt)
{
   try
   {
      NetworkLayer nl(std::shared_ptr<flexnnet::BasicLayer>(new PureLin(3, "purelin")));
      const std::vector<std::string>& xfields = nl.get_external_input_fields();

      ASSERT_EQ(xfields.size(), 0) << "Expected external fields size to be zero, got " << xfields.size() << "\n";
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NetworkLayerAddExt)
{
   try
   {
      NetworkLayer nl(std::shared_ptr<flexnnet::BasicLayer>(new PureLin(3, "purelin")));
      nl.add_external_input_field("input1");
      const std::vector<std::string>& xfields = nl.get_external_input_fields();

      ASSERT_EQ(xfields.size(), 1) << "Expected external fields size to be zero, got " << xfields.size() << "\n";
      ASSERT_EQ(xfields[0], "input1") << "Expected external field[0]=\"input1\", got " << xfields[0].c_str() << "\n";
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NNTopoEmptyDecl)
{
   try
   {
      NetworkTopology topo(sample_external_input);
   }
   catch (...)
   {
      FAIL() << "Declaration failed.";
   }
}

TEST_F(NNBuilderTestFixture, NNTopoAddSingleLayer)
{
   NetworkTopology topo(sample_external_input);

   try
   {
      topo.add_layer<PureLin>("output", 3);
      const flexnnet::NetworkLayer& layers = topo.get_layer("output");
   }
   catch (std::exception& err)
   {
      FAIL() << "Couldn't find layer \"output\" we just added.\n" << err.what();
   }
}

TEST_F(NNBuilderTestFixture, NNTopoAddTwoLayers)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);

   try
   {
      topo.add_layer<PureLin>("hidden", 3);
   }
   catch (...)
   {
      FAIL() << "Add hidden basiclayer failed.";
   }
}

TEST_F(NNBuilderTestFixture, NNTopoAddDuplicateLayer)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);

   try
   {
      topo.add_layer<PureLin>("output", 3);
      FAIL() << "Failed to detect addition of duplicate output basiclayer.";
   }
   catch (...)
   {
   }
}

TEST_F(NNBuilderTestFixture, NNTopoAddExternInput)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);

   try
   {
      topo.add_external_input_field("output", "a");
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add external input.\n" << err.what();
   }
   const std::vector<std::string>& input_fields = topo.get_external_input_fields("output");

   std::vector<std::string>::const_iterator it;
   it = std::find(input_fields.begin(), input_fields.end(), "a");
   if (it == input_fields.end())
      FAIL() << "Field \"a\" doesn't appear in input fields for \"output\".\n";

}

TEST_F(NNBuilderTestFixture, NNTopoAddMultipleExternInputs)
{
   std::set<std::string> input_set = {"a", "c"};

   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);

   try
   {
      for (auto it = input_set.begin(); it != input_set.end(); it++)
         topo.add_external_input_field("output", *it);
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add external input.\n" << err.what();
   }

   const std::vector<std::string>& ret_input_fields = topo.get_external_input_fields("output");
   std::set<std::string> ret_input_field_set;

   for (auto it = ret_input_fields.begin(); it != ret_input_fields.end(); it++)
      ret_input_field_set.insert(*it);

   if (ret_input_field_set != input_set)
      FAIL() << "Return input fields doesn't match expected\n";
}

TEST_F(NNBuilderTestFixture, NNTopoSingleForwardConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);
   topo.add_layer<PureLin>("hidden", 5);

   try
   {
      topo.add_layer_connection("output", "hidden", flexnnet::LayerConnRecord::Forward);

      // Validate connection entry
      const std::vector<flexnnet::LayerConnRecord>& layerconn = topo
         .get_activation_connections("output");
      ASSERT_EQ(layerconn.size(), 1) << "Expected output1 to have 1 input connection, got " << layerconn.size();
      ASSERT_EQ(layerconn[0].layer().name(), "hidden") << "Expected basiclayer input from \"hidden\"";
      ASSERT_EQ(layerconn[0].get_connection_type(), flexnnet::LayerConnRecord::Forward)
                     << "Expected basiclayer connection type to be Forward, got : "
                     << layerconn[0].get_connection_type();
      ASSERT_FALSE(layerconn[0].is_recurrent()) << "Expected basiclayer connection is_recurrent() to be false";
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add forward basiclayer connection.\n" << err.what();
   }
}

TEST_F(NNBuilderTestFixture, NNTopoDupForwardConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);
   topo.add_layer<PureLin>("hidden", 5);
   topo.add_layer_connection("output", "hidden", flexnnet::LayerConnRecord::Forward);

   try
   {
      topo.add_layer_connection("output", "hidden", flexnnet::LayerConnRecord::Forward);
      FAIL() << "Failed to detect duplicate forward basiclayer connection.\n";
   }
   catch (std::exception const& err)
   {
   }
}

TEST_F(NNBuilderTestFixture, NNTopoFaninForwardConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);
   topo.add_layer<PureLin>("hidden1", 5);
   topo.add_layer<PureLin>("hidden2", 2);

   std::set<std::string> conn_set = {"hidden1", "hidden2"};
   try
   {
      // Add all connections to output from conn_set
      for (auto it = conn_set.begin(); it != conn_set.end(); it++)
         topo.add_layer_connection("output", *it, flexnnet::LayerConnRecord::Forward);

      // Get the list of connections for output and verify it contains
      // all of the conn_set
      const std::vector<flexnnet::LayerConnRecord>& layerconn = topo
         .get_activation_connections("output");

      std::set<std::string> ret_set;
      for (auto i = 0; i < layerconn.size(); i++)
         ret_set.insert(layerconn[i].layer().name());

      if (ret_set != conn_set)
         FAIL() << "Return connection set doesn't match expected.\n";
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add forward basiclayer connection.\n" << err.what();
   }
}

TEST_F(NNBuilderTestFixture, NNTopoFanoutForwardConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output1", 3);
   topo.add_layer<PureLin>("output2", 5);
   topo.add_layer<PureLin>("hidden", 2);

   std::set<std::string> conn_set = {"output1", "output2"};

   try
   {
      // Add all connections to output from conn_set
      for (auto it = conn_set.begin(); it != conn_set.end(); it++)
         topo.add_layer_connection(*it, "hidden", flexnnet::LayerConnRecord::Forward);

      // Validate activation connections
      const std::vector<flexnnet::LayerConnRecord>& layerconn1 = topo
         .get_activation_connections("output1");
      ASSERT_EQ(layerconn1.size(), 1) << "Expected 1 input for output1";
      ASSERT_EQ(layerconn1[0].layer().name(), "hidden") << "Expected input from hidden for output1";

      const std::vector<flexnnet::LayerConnRecord>& layerconn2 = topo
         .get_activation_connections("output2");
      ASSERT_EQ(layerconn2.size(), 1) << "Expected 1 input for output2";
      ASSERT_EQ(layerconn2[0].layer().name(), "hidden") << "Expected input from hidden for output2";

      // Validate backprop connections
      const std::vector<flexnnet::LayerConnRecord>& bpconn1 = topo.get_backprop_connections("hidden");
      ASSERT_EQ(bpconn1.size(), 2) << "Expected 2 backprop connections for hidden";
      ASSERT_EQ(bpconn1[0].layer().name(), "output1") << "Expected backprop conn[0] from output1 for hidden";
      ASSERT_EQ(bpconn1[1].layer().name(), "output2") << "Expected backprop conn[1] from output2 for hidden";
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to get connections.\n" << err.what();
   }
}

/**
 * Test a simple two basiclayer network with a recurrent connection
 */
TEST_F(NNBuilderTestFixture, NNTopoSimpleRecurConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);
   topo.add_layer<PureLin>("hidden", 5);
   topo.add_layer_connection("output", "hidden", flexnnet::LayerConnRecord::Forward);

   try
   {
      topo.add_layer_connection("hidden", "output", flexnnet::LayerConnRecord::Recurrent);

      // Validate recurrent connection entry
      const std::vector<flexnnet::LayerConnRecord>& layerconn = topo
         .get_activation_connections("hidden");
      ASSERT_EQ(layerconn.size(), 1) << "Expected output1 to have 1 input connection, got " << layerconn.size();
      ASSERT_EQ(layerconn[0].layer().name(), "output") << "Expected basiclayer input from \"output\"";
      ASSERT_EQ(layerconn[0].get_connection_type(), flexnnet::LayerConnRecord::Recurrent)
                     << "Expected basiclayer connection type to be Recurrent, got : "
                     << layerconn[0].get_connection_type();
      ASSERT_TRUE(layerconn[0].is_recurrent()) << "Expected basiclayer connection is_recurrent() to be true";
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add recurrent basiclayer connection.\n" << err.what();
   }
}

/**
 * Test adding bad recurrent connection from two disconnected layers
 */
TEST_F(NNBuilderTestFixture, NNTopoBadRecurConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3);
   topo.add_layer<PureLin>("hidden", 5);

   try
   {
      topo.add_layer_connection("hidden", "output", flexnnet::LayerConnRecord::Recurrent);
      FAIL() << "Failed to detect bad recurrent basiclayer connection.\n";
   }
   catch (std::exception const& err)
   {
   }
}

/**
 * Test a recurrent connection in a chain of forward connected layers
 */
TEST_F(NNBuilderTestFixture, NNTopoDeepRecurConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("hidden1", 5);
   topo.add_layer<PureLin>("hidden2", 5);
   topo.add_layer<PureLin>("hidden3", 5);
   topo.add_layer<PureLin>("hidden4", 5);
   topo.add_layer<PureLin>("output", 3);

   topo.add_layer_connection("hidden2", "hidden1", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden3", "hidden2", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden4", "hidden3", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("output", "hidden4", flexnnet::LayerConnRecord::Forward);

   try
   {
      topo.add_layer_connection("hidden2", "output", flexnnet::LayerConnRecord::Recurrent);
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add recurrent basiclayer connection.\n" << err.what();
   }
}

/**
 * Test a recurrent connection in the middle of a long chain
 */
TEST_F(NNBuilderTestFixture, NNTopoMidRecurConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("hidden1", 5);
   topo.add_layer<PureLin>("hidden2", 5);
   topo.add_layer<PureLin>("hidden3", 5);
   topo.add_layer<PureLin>("hidden4", 5);
   topo.add_layer<PureLin>("output", 3);

   topo.add_layer_connection("hidden2", "hidden1", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden3", "hidden2", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden4", "hidden3", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("output", "hidden4", flexnnet::LayerConnRecord::Forward);

   try
   {
      topo.add_layer_connection("hidden2", "hidden4", flexnnet::LayerConnRecord::Recurrent);
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add recurrent basiclayer connection.\n" << err.what();
   }
}

/**
 * Test a self recurrent basiclayer
 */
TEST_F(NNBuilderTestFixture, NNTopoSelfRecurConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 5);

   try
   {
      topo.add_layer_connection("output", "output", flexnnet::LayerConnRecord::Recurrent);
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add recurrent basiclayer connection.\n" << err.what();
   }
}

/**
 * Test self-recurrent connection at end of chain for forward connections
 */
TEST_F(NNBuilderTestFixture, NNTopoEndSelfRecurConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 5);
   topo.add_layer<PureLin>("hidden", 3);

   topo.add_layer_connection("output", "hidden", flexnnet::LayerConnRecord::Forward);

   try
   {
      topo.add_layer_connection("output", "output", flexnnet::LayerConnRecord::Recurrent);
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add recurrent basiclayer connection.\n" << err.what();
   }
}

/**
 * Test self-recurrent connection in middle of chain for forward connections
 */
TEST_F(NNBuilderTestFixture, NNTopoMidSelfRecurConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("hidden1", 3);
   topo.add_layer<PureLin>("hidden2", 3);
   topo.add_layer<PureLin>("hidden3", 3);
   topo.add_layer<PureLin>("output", 5);

   topo.add_layer_connection("hidden2", "hidden1", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden3", "hidden2", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("output", "hidden3", flexnnet::LayerConnRecord::Forward);

   try
   {
      topo.add_layer_connection("hidden2", "hidden2", flexnnet::LayerConnRecord::Recurrent);
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add recurrent basiclayer connection.\n" << err.what();
   }
}

TEST_F(NNBuilderTestFixture, NNTopoSimpleLateralConn)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output1", 3);
   topo.add_layer<PureLin>("output2", 3);

   try
   {
      topo.add_layer_connection("output1", "output2", flexnnet::LayerConnRecord::Lateral);

      // Validate connection entry
      const std::vector<flexnnet::LayerConnRecord>& layerconn = topo
         .get_activation_connections("output1");
      ASSERT_EQ(layerconn.size(), 1) << "Expected output1 to have 1 input connection, got " << layerconn.size();
      ASSERT_EQ(layerconn[0].layer().name(), "output2") << "Expected basiclayer input from \"output2\"";
      ASSERT_EQ(layerconn[0].get_connection_type(), flexnnet::LayerConnRecord::Lateral)
                     << "Expected basiclayer connection type to be Lateral, got : "
                     << layerconn[0].get_connection_type();
      ASSERT_TRUE(layerconn[0].is_recurrent()) << "Expected basiclayer connection is_recurrent() to be true";
   }
   catch (std::exception const& err)
   {
      FAIL() << "Failed to add lateral basiclayer connection.\n" << err.what();
   }
}

TEST_F(NNBuilderTestFixture, NNTopoSimpleActivationOrder)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("hidden1", 3);
   topo.add_layer<PureLin>("hidden2", 3);
   topo.add_layer<PureLin>("hidden3", 3);
   topo.add_layer<PureLin>("output", 5);

   topo.add_layer_connection("hidden2", "hidden1", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden3", "hidden2", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("output", "hidden3", flexnnet::LayerConnRecord::Forward);

   const std::vector<std::shared_ptr<NetworkLayer>>& ordered_layers = topo.get_ordered_layers();

   // Check activation order
   ASSERT_EQ(ordered_layers[0]->name(), "hidden1") << "hidden1 out of order.";
   ASSERT_EQ(ordered_layers[1]->name(), "hidden2") << "hidden2 out of order.";
   ASSERT_EQ(ordered_layers[2]->name(), "hidden3") << "hidden3 out of order.";
   ASSERT_EQ(ordered_layers[3]->name(), "output") << "output out of order.";
}

TEST_F(NNBuilderTestFixture, NNTopoFaninActivationOrder)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("hidden1", 3);
   topo.add_layer<PureLin>("hidden2", 3);
   topo.add_layer<PureLin>("hidden3", 3);
   topo.add_layer<PureLin>("output", 5);

   topo.add_layer_connection("hidden3", "hidden1", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden3", "hidden2", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("output", "hidden3", flexnnet::LayerConnRecord::Forward);

   const std::vector<std::shared_ptr<NetworkLayer>>& ordered_layers = topo.get_ordered_layers();
   std::map<std::string, int> ordered_names;

   for (int i = 0; i < ordered_layers.size(); i++)
      ordered_names[ordered_layers[i]->name()] = i;

   // Check activation order
   ASSERT_GT(ordered_names["output"], ordered_names["hidden3"]) << "Fail - hidden3 expected before output.";
   ASSERT_GT(ordered_names["output"], ordered_names["hidden2"]) << "Fail - hidden2 expected before output.";
   ASSERT_GT(ordered_names["output"], ordered_names["hidden1"]) << "Fail - hidden1 expected before output.";
   ASSERT_GT(ordered_names["hidden3"], ordered_names["hidden2"]) << "Fail - hidden2 expected before hidden3.";
   ASSERT_GT(ordered_names["hidden3"], ordered_names["hidden1"]) << "Fail - hidden1 expected before hidden3.";
}

TEST_F(NNBuilderTestFixture, NNTopoActOrderWRecur)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("hidden1", 3);
   topo.add_layer<PureLin>("hidden2", 3);
   topo.add_layer<PureLin>("hidden3", 3);
   topo.add_layer<PureLin>("output", 5);

   topo.add_layer_connection("hidden2", "hidden1", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden3", "hidden2", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("output", "hidden3", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden3", "output", flexnnet::LayerConnRecord::Recurrent);

   const std::vector<std::shared_ptr<NetworkLayer>>& ordered_layers = topo.get_ordered_layers();

   // Check activation order
   ASSERT_EQ(ordered_layers[0]->name(), "hidden1") << "hidden1 out of order.";
   ASSERT_EQ(ordered_layers[1]->name(), "hidden2") << "hidden2 out of order.";
   ASSERT_EQ(ordered_layers[2]->name(), "hidden3") << "hidden3 out of order.";
   ASSERT_EQ(ordered_layers[3]->name(), "output") << "output out of order.";
}

TEST_F(NNBuilderTestFixture, NNTopoFaninActOrderRandConnOrder)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("hidden1", 3);
   topo.add_layer<PureLin>("hidden2", 3);
   topo.add_layer<PureLin>("hidden3", 3);
   topo.add_layer<PureLin>("output", 5);

   topo.add_layer_connection("output", "hidden3", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden2", "hidden1", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("hidden3", "hidden2", flexnnet::LayerConnRecord::Forward);

   const std::vector<std::shared_ptr<NetworkLayer>>& ordered_layers = topo.get_ordered_layers();

   // Check activation order
   ASSERT_EQ(ordered_layers[0]->name(), "hidden1") << "hidden1 out of order.";
   ASSERT_EQ(ordered_layers[1]->name(), "hidden2") << "hidden2 out of order.";
   ASSERT_EQ(ordered_layers[2]->name(), "hidden3") << "hidden3 out of order.";
   ASSERT_EQ(ordered_layers[3]->name(), "output") << "output out of order.";
}

TEST_F(NNBuilderTestFixture, NNTopoNNOutputLayers)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output", 3, true);
   topo.add_layer<PureLin>("hidden1", 5);
   topo.add_layer<PureLin>("hidden2", 2);

   topo.add_layer_connection("output", "hidden1", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("output", "hidden2", flexnnet::LayerConnRecord::Forward);

   // Validate activation connections
   const std::vector<std::shared_ptr<NetworkLayer>>& olayers = topo.get_output_layers();
   ASSERT_EQ(olayers.size(), 1) << "Expected 1 output basiclayer";
   ASSERT_EQ(olayers[0]->name(), "output") << "Expected input from hidden for output1";
}

TEST_F(NNBuilderTestFixture, NNTopoFanoutOutputLayers)
{
   NetworkTopology topo(sample_external_input);
   topo.add_layer<PureLin>("output1", 3, true);
   topo.add_layer<PureLin>("output2", 5, true);
   topo.add_layer<PureLin>("hidden", 2);

   topo.add_layer_connection("output1", "hidden", flexnnet::LayerConnRecord::Forward);
   topo.add_layer_connection("output2", "hidden", flexnnet::LayerConnRecord::Forward);

   // Validate activation connections
   const std::vector<std::shared_ptr<NetworkLayer>>& olayers = topo.get_output_layers();
   ASSERT_EQ(olayers.size(), 2) << "Expected 2 output basiclayer";
   ASSERT_EQ(olayers[0]->name(), "output1") << "Expected output_layer[0] to be \"output1\"";
   ASSERT_EQ(olayers[1]->name(), "output2") << "Expected output_layer[1] to be \"output2\"";
}
#endif //_NNBUILDERTESTFIXTURE_H_
