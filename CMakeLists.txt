cmake_minimum_required(VERSION 3.5)
project(parallelSBMP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

include_directories(
    include 
    ${EIGEN3_INCLUDE_DIRS} 
    ${BOOST_INCLUDE_DIRS}  
    ${CMAKE_SOURCE_DIR}/src 
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# Add the source files for your project
set(SOURCE_FILES
    src/agent/Agent.cpp
    src/state/State.cpp
    src/planners/Planner.cu
    src/planners/NaivePlanner.cu
    src/planners/CostPropPlanner.cu
    src/planners/KGMT.cu
    src/helper/helper.cu
    src/occupancyMaps/OccupancyGrid.cu
)

# Add a library target for your non-CUDA code
add_library(parallelSBMP_lib ${SOURCE_FILES})

# Add your CUDA executable
add_executable(gpu_demo1 demos/gpu_demo1.cu)

# Link the library and other dependencies with the CUDA executable
target_link_libraries(gpu_demo1
    parallelSBMP_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

# Ensure CUDA separable compilation
set_target_properties(gpu_demo1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)