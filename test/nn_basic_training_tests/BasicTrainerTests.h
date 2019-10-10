//
// Created by kfedrick on 6/25/19.
//

#ifndef _BASICTRAINERTESTS_H_
#define _BASICTRAINERTESTS_H_

#include "gtest/gtest.h"

#include <evaluators/include/FuncApproxEvaluator.h>
#include <datasets/include/DataSet2.h>
#include "evaluators/include/RMSError.h"
#include "FATrainer.h"
#include "BasicTrainer.h"
#include "BackpropAlgo.h"

using flexnnet::Datum;
using flexnnet::Exemplar;
using flexnnet::NeuralNet;
using flexnnet::RMSError;
using flexnnet::FuncApproxEvaluator;
using flexnnet::FATrainer;
using flexnnet::BasicTrainer;
using flexnnet::BackpropAlgo;


/**
 * Test basic methods common to all trainers
 *
 * @tparam T - trainer type
 */
template<typename T>
class BasicTrainerTests : public ::testing::Test
{
public:
   virtual void SetUp ();

   virtual void TearDown ()
   {}

protected:
   std::unique_ptr<NeuralNet<Datum,Datum>> nnet;

   flexnnet::DataSet<Datum,Datum,Exemplar> trnset;
   flexnnet::DataSet<Datum,Datum,Exemplar> vldset;
   flexnnet::DataSet<Datum,Datum,Exemplar> tstset;
};

template<typename T>
void BasicTrainerTests<T>::SetUp()
{
   // Create neural network with no layers
   std::vector<std::shared_ptr<flexnnet::NetworkLayer>> layers;
   nnet.reset( new NeuralNet<Datum,Datum>(layers, false) );

   // Add entries to test data set_weights.
   Datum adatum1;
   adatum1.insert ("in1", {0,0,0});

   Datum adatum2;
   adatum2.insert ("in2", {0,0,0});

   Datum adatum3;
   adatum3.insert ("in3", {0,0,0});
   adatum3.insert ("in4", {0,0,1});

   flexnnet::Exemplar<Datum,Datum> exemplar1(adatum1,adatum2);
   flexnnet::Exemplar<Datum,Datum> exemplar2(adatum1,adatum3);

   trnset.insert(exemplar1);
   trnset.insert(exemplar2);
}



TYPED_TEST_CASE_P (BasicTrainerTests);

//typedef ::testing::Types<FATrainer<Datum,Datum,Exemplar,RMSError,FuncApproxEvaluator>> MyTypes;
typedef ::testing::Types<BasicTrainer<Datum, Datum, Exemplar, BackpropAlgo, FuncApproxEvaluator, RMSError>> MyTypes;


#endif //_TEST_NN_ACTIVATION_H_
