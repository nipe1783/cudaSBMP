#pragma once
#include "helper/helper.cuh"
#include <Eigen/Core>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

__host__ void sssp(int numNodes, int numEdges, float* d_edgeCosts_ptr, int* d_fromNodes_ptr, int* d_toNodes_ptr, float* d_dist_ptr,
                   bool* d_finished_ptr, int* d_predNode_ptr, int r1_initial, int r1_goal);

__global__ void sssp_kernel(int numEdges, int numEdgesPerThread, float* edgeCosts, int* fromNodes, int* toNodes, float* dist, bool* finished, int* predNode);

/************************/
/* dijkstraGPU FUNCTION */
/************************/
__host__ void dijkstraGPU(const int sourceVertex, const int numVertices, const int numEdges,
                          int* d_vertexArray, int* d_edgeArray, float* d_weightArray, bool* d_finalizedVertices, float* d_shortestDistances,
                          float* d_updatingShortestDistances, bool* h_finalizedVertices);

/***************************/
/* DIJKSTRA GPU KERNEL #1  */
/***************************/
__global__ void Kernel1(const int* __restrict__ vertexArray, const int* __restrict__ edgeArray, const float* __restrict__ weightArray,
                        bool* __restrict__ finalizedVertices, float* __restrict__ shortestDistances, float* __restrict__ updatingShortestDistances,
                        const int numVertices, const int numEdges);

/**************************/
/* DIJKSTRA GPU KERNEL #2 */
/**************************/
__global__ void Kernel2(const int* __restrict__ vertexArray, const int* __restrict__ edgeArray, const float* __restrict__ weightArray,
                        bool* __restrict__ finalizedVertices, float* __restrict__ shortestDistances, float* __restrict__ updatingShortestDistances,
                        const int numVertices);
