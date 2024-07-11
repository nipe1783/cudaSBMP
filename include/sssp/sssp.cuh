#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include <ctime>
#include <cub/cub.cuh>
#include "helper/helper.cuh"

__host__  void sssp(
    int numNodes, 
    int numEdges, 
    float* d_edgeCosts_ptr, 
    int *d_fromNodes_ptr, 
    int *d_toNodes_ptr, 
    float* d_dist_ptr, 
    bool* d_finished_ptr,
    int* d_predNode_ptr,
    int r1_initial,
    int r1_goal);

__global__ void sssp_kernel(
    int numEdges, 
    int numEdgesPerThread, 
    float* edgeCosts, 
    int *fromNodes, 
    int *toNodes, 
    float* dist,
    bool* finished,
    int* predNode);