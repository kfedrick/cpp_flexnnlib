//
// Created by kfedrick on 6/25/19.
//

#ifndef _TEMPLATE_TEST_EXAMPLE_H_
#define _TEMPLATE_TEST_EXAMPLE_H_

#include "gtest/gtest.h"

/**
 * Example of a type template google test fixture
 *
 * @tparam T - trainer type
 */
template<typename T>
class TemplateTestExample : public ::testing::Test
{
public:
   virtual void SetUp ()
   {}
   virtual void TearDown ()
   {}

   // Fixture specific methods go here - available to all tests using this fixture
};

// Macro to initialize the code needed to turn this into a parameterized fixture
TYPED_TEST_CASE_P (TemplateTestExample);

// Specify the types that the framework should instantiate for this fixture
typedef ::testing::Types</* specify here */> MyTypes;


#endif //_TEMPLATE_TEST_EXAMPLE_H_
