cmake_minimum_required(VERSION 3.13)
project(cpp_flexnnlib)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions")

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
        include/flexnnet.h
        util/include/Array.h
        util/include/NamedObject.h
        layers/include/BasicLayer.h layers/BasicLayer.cpp
        util/include/LayerWeights.h util/LayerWeights.cpp
        util/include/LayerState.h
        layers/transfer_functions/include/PureLin.h layers/transfer_functions/PureLin.cpp
        layers/transfer_functions/include/LogSig.h layers/transfer_functions/LogSig.cpp
        layers/transfer_functions/include/RadBas.h layers/transfer_functions/RadBas.cpp
        layers/transfer_functions/include/TanSig.h layers/transfer_functions/TanSig.cpp
        layers/transfer_functions/include/SoftMax.h layers/transfer_functions/SoftMax.cpp
        util/include/Array2D.h
        util/include/Array3D.h
        network/include/ExternalInputRecord.h network/ExternalInputRecord.cpp
        layers/transfer_functions/include/NetSumLayer.h layers/transfer_functions/NetSumLayer.cpp
        layers/transfer_functions/include/EuclideanDistLayer.h layers/transfer_functions/EuclideanDistLayer.cpp
        trainers/include/TrainerConfig.h
        datasets/include/DataSet.h
        trainers/include/TrainingRecord.h
        evaluators/include/Evaluator.h
        evaluators/include/RMSEFitnessFunc.h
        datasets/include/CartesianCoord.h datasets/CartesianCoord.cpp
        datasets/include/NNetIOInterface.h
        util/include/JSONStream.h util/JSONStream.cpp
        util/include/Globals.h util/Globals.cpp
        datasets/stream/CartesianCoordStream.cpp
        datasets/stream/include/DataSetStream.h
        datasets/include/ValarrayMap.h datasets/ValarrayMap.cpp
        evaluators/include/BaseFitnessFunc.h evaluators/BaseFitnessFunc.cpp
        datasets/stream/ValarrayMapStream.cpp
        trainers/include/DeltaBarDeltaLearningRate.h
        trainers/include/TrainingReport.h
        util/include/URandArray2DInitializer.h
        util/include/NRandArray2DInitializer.h
        util/LayerState.cpp
        network/NetworkLayer.cpp
        network/include/NetworkLayer.h
        network/include/LayerConnRecord.h
        network/include/NetworkLayerImpl.h
        network/NeuralNetBuilder.cpp
        network/include/NeuralNetBuilder.h
        network/NeuralNetTopology.cpp
        network/include/NeuralNetTopology.h
        network/BaseNetworkLayer.cpp
        network/include/BaseNetworkLayer.h
        network/BaseNeuralNet.cpp
        network/include/BaseNeuralNet.h
        network/include/NeuralNet.h
        trainers/include/BaseTrainer.h
        trainers/include/SuperviseTrainingAlgo.h
        trainers/include/LearningRatePolicy.h
        trainers/include/ConstantLearningRate.h
        )

target_include_directories(flexneuralnet PUBLIC
        actor_critic
        actor_critic/include
        datasets
        datasets/include
        datasets/stream
        datasets/stream/include
        include
        layers
        layers/include
        layers/transfer_functions/include
        network
        network/include
        evaluators
        evaluators/include
        trainers
        trainers/include
        util
        util/include
        rapidjson
        picojson
        )

enable_testing()
add_subdirectory(test)