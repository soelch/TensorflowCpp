cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)


set(cpps Main.cpp NeuralNet.cpp ReadCSV.cpp TensorCalc.cpp)

add_executable(example ${cpps})
#SET(GCC_COVERAGE_COMPILE_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

#SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(example ${Protobuf_LIBRARIES})
find_package(TensorflowCC REQUIRED)
target_link_libraries(example TensorflowCC::TensorflowCC)

# link cuda if it is available
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(example ${CUDA_LIBRARIES})
endif()
