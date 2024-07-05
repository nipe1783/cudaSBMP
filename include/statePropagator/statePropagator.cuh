#pragma once
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <chrono>
#include <ctime>
#include <cub/cub.cuh>
#include <filesystem>

__device__ bool propagateAndCheck(
    float* x0, 
    float* x1, 
    int numDisc, 
    float agentLength, 
    curandState* state);