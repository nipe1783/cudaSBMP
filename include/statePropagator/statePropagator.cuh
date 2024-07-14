#pragma once
#include <curand_kernel.h>
#include "collisionCheck/collisionCheck.cuh"
#include "occupancyMaps/OccupancyGrid.cuh"

__device__ bool propagateAndCheck(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount);
__device__ bool propagateAndCheckUnicycle(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount);
__device__ bool propagateAndCheckDubins(float* x0, float* x1, curandState* seed, float* obstacles, int obstaclesCount);

typedef bool (*PropagateAndCheckFunc)(float*, float*, curandState*, float*, int);

/***************************/
/* GET PROPAGATION FUNCTION */
/***************************/
// --- Determins which dynamic model to use. ---
__device__ PropagateAndCheckFunc getPropagateAndCheckFunc();