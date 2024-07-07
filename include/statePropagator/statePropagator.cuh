#pragma once
#include <curand_kernel.h>
#include "collisionCheck/collisionCheck.cuh"

__device__ bool propagateAndCheck(
    float* x0, 
    float* x1, 
    int numDisc, 
    float agentLength, 
    curandState* state,
    float* obstacles,
    int obstaclesCount,
    float width,
    float height);