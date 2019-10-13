cmake_minimum_required(VERSION 3.13)
project(cpp_flexnnlib)

set(CMAKE_CXX_STANDARD 14)
# set_weights(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

include_directories("${PROJECT_SOURCE_DIR}")

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
        ${CMAKE_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif ()

add_executable(sandbox sandbox/doit.cpp)


add_library(flexneuralnet
        datasets/include/Exemplar.h
        include/flexnnet.h
        network/include/SumSquaredError.h
        network/SumSquaredError.cpp
        util/include/Array.h
        util/include/ArrayInitializer.h
        util/include/NamedObject.h
        util/include/URandArrayInitializer.h
        util/ArrayInitializer.cpp
        util/URandArrayInitializer.cpp
        layers/include/BasicLayer.h
        layers/BasicLayer.cpp
        util/LayerWeights.cpp
        util/include/LayerWeights.h
        network/BasicNeuralNet.cpp
        network/include/BasicNeuralNet.h
        util/LayerDerivatives.cpp
        util/include/LayerDerivatives.h
        util/include/LayerState.h
        layers/transfer_functions/PureLin.cpp
        layers/transfer_functions/include/PureLin.h
        util/JSONEncoder.cpp
        util/include/JSONEncoder.h
        layers/transfer_functions/LogSig.cpp
        layers/transfer_functions/include/LogSig.h
        layers/transfer_functions/RadBas.cpp
        layers/transfer_functions/include/RadBas.h
        layers/transfer_functions/TanSig.cpp
        layers/transfer_functions/include/TanSig.h
        layers/serializers/include/LayerSerializer.h
        layers/serializers/PureLinSerializer.cpp
        layers/serializers/BasicLayerSerializer.cpp
        layers/serializers/include/BasicLayerSerializer.h
        layers/serializers/TanSigSerializer.cpp
        layers/serializers/RadBasSerializer.cpp
        layers/serializers/LogSigSerializer.cpp
        layers/transfer_functions/SoftMax.cpp
        layers/transfer_functions/include/SoftMax.h
        layers/serializers/SoftMaxSerializer.cpp
        network/BasicNeuralNetFactory.cpp
        network/include/BasicNeuralNetFactory.h
        datasets/Datum.cpp
        datasets/include/Datum.h
        util/include/Array2D.h
        util/include/Array3D.h
        layers/include/NetworkLayer.h
        network/LayerConnRecord.cpp
        network/include/LayerConnRecord.h
        network/ExternalInputRecord.cpp
        network/include/ExternalInputRecord.h
        network/BasicNeuralNetSerializer.cpp
        network/include/BasicNeuralNetSerializer.h
        network/BasicNeuralNetDeserializer.cpp
        layers/transfer_functions/NetSumLayer.cpp
        layers/transfer_functions/include/NetSumLayer.h
        layers/transfer_functions/EuclideanDistLayer.cpp
        layers/transfer_functions/include/EuclideanDistLayer.h
        trainers/include/TrainerConfig.h
        network/LayerInput.cpp
        network/include/LayerInput.h
        network/NetworkOutput.cpp
        network/include/NetworkOutput.h
        network/NetworkLayer.cpp
        trainers/include/PerformanceMetrics.h
        datasets/include/DataSet.h
        datasets/include/Exemplar.h
        datasets/include/Episode.h
        network/include/NeuralNet.h
        trainers/include/TrainerUtils.h
        trainers/FATrainer.cpp
        trainers/include/FATrainer.h
        evaluators/include/BasicEvaluator.h
        network/include/NeuralNetSerializer.h
        datasets/include/VectorConcatenator.h
        datasets/include/Vectorizable.h
        datasets/include/A.h
        datasets/include/VectorizableSet.h
        evaluators/include/ActorCriticEvaluator.h
        evaluators/include/BasicEvalConfig.h
        trainers/include/TDTrainerConfig.h
        evaluators/include/TDEvaluatorConfig.h
        trainers/include/ActorCriticTrainer.h
        evaluators/include/RMSError.h
        trainers/include/TrainingRecord.h
        trainers/include/flexnnet_trainers.h
        util/include/flexnnet_utils.h
        layers/include/flexnnet_layers.h
        network/include/flexnnet_networks.h
        trainers/include/BasicTrainer.h
        trainers/include/BackpropAlgo.h
        util/include/NetworkError.cpp
        util/include/NetworkError.h
        datasets/include/VectorMap.h datasets/include/NamedValarray.h datasets/include/VectorizableMap.h)

target_include_directories(flexneuralnet PUBLIC
        actor_critic
        actor_critic/include
        datasets
        datasets/include
        include
        layers
        layers/include
        layers/serializers/include
        layers/transfer_functions/include
        network
        network/include
        trainers
        trainers/include
        util
        util/include
        rapidjson)

#enable_testing()
add_subdirectory(test)