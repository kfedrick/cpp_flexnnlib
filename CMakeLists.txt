cmake_minimum_required(VERSION 3.13)
project(cpp_neuralnetlib)

set(CMAKE_CXX_STANDARD 14)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

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
endif()



add_library(flexneuralnet
        actor_critic/include/Action.h
        actor_critic/include/ActorCriticNet.h
        actor_critic/include/ActorCriticNet2.h
        actor_critic/include/ActorCriticNet3.h
        actor_critic/include/ActorCriticOutput.h
        actor_critic/include/ActorCriticTrainer.h
        actor_critic/include/ActorCriticTrainer2.h
        actor_critic/include/ActorCriticTrainer3.h
        actor_critic/include/ActorNet.h
        actor_critic/include/ActorNet2.h
        actor_critic/include/ActorNet3.h
        actor_critic/include/AdaptiveComponent.h
        actor_critic/include/AdaptiveCriticNet.h
        actor_critic/include/AdaptiveCriticNet3.h
        actor_critic/include/Critic.h
        actor_critic/include/EnvironmentSimulator.h
        actor_critic/include/GreedyActorCriticNet.h
        actor_critic/include/GreedyActorCriticTrainer.h
        actor_critic/include/GreedyActorNet.h
        actor_critic/include/PBActorCriticNet.h
        actor_critic/include/PBActorCriticTrainer.h
        actor_critic/include/PBActorNet.h
        actor_critic/Action.cpp
        actor_critic/ActorCriticNet.cpp
        actor_critic/ActorCriticNet2.cpp
        actor_critic/ActorCriticNet3.cpp
        actor_critic/ActorCriticOutput.cpp
        actor_critic/ActorCriticTrainer.cpp
        actor_critic/ActorCriticTrainer2.cpp
        actor_critic/ActorCriticTrainer3.cpp
        actor_critic/ActorNet.cpp
        actor_critic/ActorNet2.cpp
        actor_critic/ActorNet3.cpp
        actor_critic/AdaptiveCriticNet.cpp
        actor_critic/AdaptiveCriticNet3.cpp
        actor_critic/GreedyActorCriticNet.cpp
        actor_critic/GreedyActorCriticTrainer.cpp
        actor_critic/GreedyActorNet.cpp
        actor_critic/PBActorCriticNet.cpp
        actor_critic/PBActorCriticTrainer.cpp
        actor_critic/PBActorNet.cpp
        datasets/include/DataSet.h
        datasets/include/Exemplar.h
        datasets/include/Pattern.h
        datasets/include/PatternSequence.h
        datasets/Pattern.cpp
        datasets/PatternSequence.cpp
        include/neuralnet.h
        layers/include/BaseLayer.h
        layers/include/ConnectionMap.h
        layers/include/HvMap.h
        layers/include/Layer.h
        layers/include/LogSig.h
        layers/include/NetDist.h
        layers/include/NetInputFunctor.h
        layers/include/NetSum.h
        layers/include/PureLin.h
        layers/include/RadBas.h
        layers/include/SoftMax.h
        layers/include/TanSig.h
        layers/include/TransferFunctor.h
        layers/BaseLayer.cpp
        layers/ConnectionMap.cpp
        layers/HvMap.cpp
        layers/LogSig.cpp
        layers/NetDist.cpp
        layers/NetSum.cpp
        layers/PureLin.cpp
        layers/RadBas.cpp
        layers/SoftMax.cpp
        layers/TanSig.cpp
        network/include/BaseNeuralNet.h
        network/include/BaseTrainer.h
        network/include/ConstantLearningRate.h
        network/include/DeltaBarDeltaLearningRate.h
        network/include/GradientDescentTrainer.h
        network/include/LearningRatePolicy.h
        network/include/OutputErrorFunctor.h
        network/include/SumSquaredError.h
        network/include/TD2Trainer.h
        network/include/TDC0aTrainer.h
        network/include/TDC0Trainer.h
        network/include/TDCNeuralNet.h
        network/include/TDCTrainer.h
        network/include/TDCTrainer2.h
        network/include/TDTrainer.h
        network/include/TemporalDiffTrainer.h
        network/include/TemporalDiffTrainer2.h
        network/include/TemporalDiffTrainer3.h
        network/include/TrainingRecord.h
        network/BaseNeuralNet.cpp
        network/BaseTrainer.cpp
        network/OutputErrorFunctor.cpp
        network/SumSquaredError.cpp
        network/TDCNeuralNet.cpp
        sandbox/LevMarqTrainer.h
        util/include/Array.h
        util/include/ArrayInitializer.h
        util/include/NamedObject.h
        util/include/NetworkWeightsData.h
        util/include/Parameters.h
        util/include/URandArrayInitializer.h
        util/ArrayInitializer.cpp
        util/NamedObject.cpp
        util/NetworkWeightsData.cpp
        util/Parameters.cpp
        util/URandArrayInitializer.cpp)

target_include_directories(flexneuralnet PUBLIC
        actor_critic
        actor_critic/include
        datasets
        datasets/include
        include
        layers
        layers/include
        network
        network/include
        sandbox
        util
        util/include)