//
// Created by kfedrick on 6/25/19.
//

#include "TemplateTestFixtureExample.h"

//                fixture name   Test name
//                      |            |
//                      V            V
TYPED_TEST_P (TemplateTestExample, Simple)
{
   std::cout << "\nBasicTrainerTests::Simple\n";

   TypeParam basic_trainer;
}

//                Macro to register each test
//
//                           fixture name     test names
//                                 |              |
//                                 V              V
REGISTER_TYPED_TEST_CASE_P(TemplateTestExample, Simple /* ... */);
INSTANTIATE_TYPED_TEST_CASE_P(My, TemplateTestExample, MyTypes);

int main (int argc, char **argv)
{
   ::testing::InitGoogleTest (&argc, argv);

   return RUN_ALL_TESTS ();
}