include_directories(".." "include")

enable_testing()

add_executable(baselayer_activation_tests baselayer_activation_tests/test_main.cpp
        baselayer_activation_tests/LayerActivationTestCase.h
        CommonTestFixtureFunctions.cpp include/CommonTestFixtureFunctions.h
        baselayer_activation_tests/test_purelin_activation.cpp
        baselayer_activation_tests/test_tansig_activation.cpp
        baselayer_activation_tests/test_logsig_activation.cpp
        baselayer_activation_tests/test_radbas_activation.cpp
        baselayer_activation_tests/test_softmax_activation.cpp esimulator_tests/DerbySim.h esimulator_tests/SteeringAction.h esimulator_tests/DerbyTests.h)
target_link_libraries(baselayer_activation_tests gtest flexneuralnet)
add_test(NAME baselayer_activation_tests COMMAND test_main)

add_executable(networklayer_tests networklayer_tests/NNBuilderTestFixture.h
        CommonTestFixtureFunctions.cpp include/CommonTestFixtureFunctions.h
        networklayer_tests/test_main.cpp
        )
target_link_libraries(networklayer_tests gtest flexneuralnet)
add_test(NAME networklayer_tests COMMAND test_main)

add_executable(test_nn_activation nn_activation_tests/test_nn_activation.cpp
        nn_activation_tests/test_single_layer_nnet_activation.cpp nn_activation_tests/test_single_layer_nnet_activation.h
        nn_activation_tests/test_fanin_ffnnet_activation.cpp nn_activation_tests/test_fanin_ffnnet_activation.h
        nn_activation_tests/test_fanout_ffnnet_activation.cpp nn_activation_tests/test_fanout_ffnnet_activation.h
        TestLayer.cpp)
target_link_libraries(test_nn_activation gtest flexneuralnet)
add_test(NAME nn_activation_tests COMMAND test_nn_activation)

add_executable(test_basenn_activation
        include/CommonTestFixtureFunctions.h CommonTestFixtureFunctions.cpp
        basenn_activation_tests/SingleLayerNNActivationTestFixture.h basenn_activation_tests/SingleLayerNNActivationTestFixture.cpp
        basenn_activation_tests/FaninNNActivationTestFixture.h basenn_activation_tests/FaninNNActivationTestFixture.cpp
        basenn_activation_tests/FanoutNNActivationTestFixture.h basenn_activation_tests/FanoutNNActivationTestFixture.cpp
        basenn_activation_tests/test_main.cpp)
target_link_libraries(test_basenn_activation gtest flexneuralnet)
add_test(NAME test_basenn_activation COMMAND test_main)

add_executable(dataset_tests dataset_tests/test_main.cpp
        include/CommonTestFixtureFunctions.h CommonTestFixtureFunctions.cpp
        )
target_link_libraries(dataset_tests gtest flexneuralnet)
add_test(NAME dataset_tests COMMAND test_main)

add_executable(evaluator_tests evaluator_tests/test_main.cpp
        include/CommonTestFixtureFunctions.h CommonTestFixtureFunctions.cpp
        nn_basic_training_tests/SimpleBinaryClassifierDataSet.h evaluator_tests/MockNN.h nn_basic_training_tests/ClassifierTrainingTests.h)
target_link_libraries(evaluator_tests gtest flexneuralnet)
add_test(NAME evaluator_tests COMMAND test_main)

add_executable(test_nn_basic_training
        include/CommonTestFixtureFunctions.h CommonTestFixtureFunctions.cpp
        nn_basic_training_tests/MockNN.h
        nn_basic_training_tests/SimpleBinaryClassifierDataSet.h
        nn_basic_training_tests/BoundedRandomWalkDataSet.h
        nn_basic_training_tests/SupervisedTrainerTestFixture.h
        nn_basic_training_tests/TrainerConfigTests.h
        nn_basic_training_tests/test_main.cpp
        nn_basic_training_tests/ClassifierTrainingTests.h
        )
target_link_libraries(test_nn_basic_training gtest flexneuralnet)
add_test(NAME test_nn_basic_training COMMAND test_main)

add_executable(esimulator_tests esimulator_tests/test_main.cpp
        include/CommonTestFixtureFunctions.h CommonTestFixtureFunctions.cpp
        esimulator_tests/DerbySim.h esimulator_tests/SteeringAction.h)
target_link_libraries(esimulator_tests gtest flexneuralnet)
add_test(NAME esimulator_tests COMMAND test_main)

add_executable(actorcritic_tests actorcritic_tests/test_main.cpp
        include/CommonTestFixtureFunctions.h CommonTestFixtureFunctions.cpp
        actorcritic_tests/LabeledFeatureSet.h
        actorcritic_tests/TestAction.h actorcritic_tests/TestActionFeature.h
        actorcritic_tests/BasicACTests.h actorcritic_tests/ACActivationTests.h)
target_link_libraries(actorcritic_tests gtest flexneuralnet)
add_test(NAME actorcritic_tests COMMAND test_main)

add_executable(nnio_tests nnio_tests/test_main.cpp
        include/CommonTestFixtureFunctions.h CommonTestFixtureFunctions.cpp
        nnio_tests/FeatureSetTests.h nnio_tests/LabeledFeatureSet.h
        nnio_tests/TestAction.h nnio_tests/TestActionFeature.h)
target_link_libraries(nnio_tests gtest flexneuralnet)
add_test(NAME nnio_tests COMMAND test_main)