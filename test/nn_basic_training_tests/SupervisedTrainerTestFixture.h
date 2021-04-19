//
// Created by kfedrick on 6/25/19.
//

#ifndef _SUPERVISEDTRAINERTESTFIXTURE_H_
#define _SUPERVISEDTRAINERTESTFIXTURE_H_

#include "gtest/gtest.h"
#include <NeuralNet.h>
#include <DataSet.h>
#include <CommonTestFixtureFunctions.h>
#include <ValarrayMap.h>
#include <Exemplar.h>

using flexnnet::NeuralNet;
using flexnnet::Exemplar;


class SupervisedTrainerTestFixture : public CommonTestFixtureFunctions, public ::testing::Test
{
public:
   virtual void SetUp();

   virtual void TearDown();

protected:
   std::unique_ptr<NeuralNet<flexnnet::ValarrayMap, flexnnet::ValarrayMap>> nnet;

   flexnnet::DataSet<flexnnet::ValarrayMap, flexnnet::ValarrayMap, Exemplar> trnset;
};

void SupervisedTrainerTestFixture::SetUp()
{

}

void SupervisedTrainerTestFixture::TearDown()
{}

#endif //_SUPERVISEDTRAINERTESTFIXTURE_H_
