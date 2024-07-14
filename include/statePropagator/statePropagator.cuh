#pragma once
#include <curand_kernel.h>
#include "collisionCheck/collisionCheck.cuh"
#include "occupancyMaps/OccupancyGrid.cuh"

__device__ bool propagateAndCheck(float* x0, float* x1, int numDisc, float agentLength, curandState* state, float* obstacles,
                                  int obstaclesCount, float width, float height);

__device__ bool
propagateAndCheck_gb(float* x0, float* x1, int numDisc, float agentLength, curandState* state, float* obstacles, int obstaclesCount,
                     float width, float height, int* selR1Edge, int* valR1Edge, float R1Size, int N, int* hashTable, int tableSize);