#include "planners/RRT.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "agent/Agent.h"

void RRT::plan(float* start, float* goal){
    
    float* samples = nullptr;
    float* controls = nullptr;
    generateRandomTreeV2(start, 100, &samples);
}