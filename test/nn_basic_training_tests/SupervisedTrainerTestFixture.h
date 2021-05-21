//
// Created by kfedrick on 6/25/19.
//

#ifndef _SUPERVISEDTRAINERTESTFIXTURE_H_
#define _SUPERVISEDTRAINERTESTFIXTURE_H_

#include "gtest/gtest.h"
#include <NeuralNet.h>
#include <DataSet.h>
#include <CommonTestFixtureFunctions.h>
#include <FeatureVector.h>
#include <Exemplar.h>

using flexnnet::NeuralNet;
using flexnnet::Exemplar;


class SupervisedTrainerTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   virtual void SetUp();

   virtual void TearDown();

protected:
   std::unique_ptr<NeuralNet<flexnnet::FeatureVector, flexnnet::FeatureVector>> nnet;

   flexnnet::DataSet<flexnnet::FeatureVector, flexnnet::FeatureVector, Exemplar> trnset;
};

void SupervisedTrainerTestFixture::SetUp()
{

}

void SupervisedTrainerTestFixture::TearDown()
{}

#endif //_SUPERVISEDTRAINERTESTFIXTURE_H_
