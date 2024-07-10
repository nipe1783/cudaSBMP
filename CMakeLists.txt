cmake_minimum_required(VERSION 3.5)
project(parallelSBMP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(cudf REQUIRED)
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
    ${YAML_CPP_INCLUDE_DIR}
    ${CUDF_INCLUDE_DIRS}
)

# Add the source files for your project
set(SOURCE_FILES
    src/agent/Agent.cpp
    src/state/State.cpp
    src/planners/KGMT.cu
    src/planners/GoalBiasedKGMT.cu
    src/helper/helper.cu
    src/occupancyMaps/OccupancyGrid.cu
    src/collisionCheck/collisionCheck.cu
    src/statePropagator/statePropagator.cu
)

# Add a library target for your non-CUDA code
add_library(parallelSBMP_lib ${SOURCE_FILES})

# Add your CUDA executable
add_executable(main demos/main.cu)

# Link the library and other dependencies with the CUDA executable
target_link_libraries(main
    parallelSBMP_lib
    ${CUDA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${CUDF_LIBRARIES}
    yaml-cpp
    Python3::Python
    Python3::NumPy
)

# Ensure CUDA separable compilation
set_target_properties(parallelSBMP_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Suppress specific CUDA warnings and enable detailed information
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -diag-suppress=20012,20013,20014,20015 --expt-relaxed-constexpr --ptxas-options=-v")

# Set CUDA architectures
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 52)
set_property(TARGET parallelSBMP_lib PROPERTY CUDA_ARCHITECTURES 52)

# Suppress all warnings from Eigen
add_definitions(-DEIGEN_NO_DEBUG)

# Additional configurations for suppressing warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unused-function")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4244 /wd4267")
endif()

# Suppress warnings from CUDA headers
add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
add_definitions(-D_FORCE_INLINES)
