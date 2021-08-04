cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

#either LoadModel.cpp or Standalone.cpp
set(cpps Standalone.cpp NeuralNet.cpp ReadCSV.cpp TensorCalc.cpp)

add_executable(example ${cpps})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(example ${Protobuf_LIBRARIES})
find_package(TensorflowCC REQUIRED)
target_link_libraries(example TensorflowCC::TensorflowCC)


