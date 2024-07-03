#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "planners/Planner.cuh"
#include "planners/NaivePlanner.cuh"
#include "planners/CostPropPlanner.cuh"
#include "occupancyMaps/OccupancyGrid.cuh"
#include "planners/KGMT.cuh"

int main(void) {
    int sampleDim = 7;
    float width = 10.0;
    float height = 10.0;
    int N = 16;
    int n = 16;
    int numIterations = 1;
    int maxTreeSize = 50000;
    int maxSampleSize = 20000;
    int numDisc = 100;
    float agentLength = 1.0;

    KGMT kgmt(width, height, N, n, numIterations, maxTreeSize, maxSampleSize, numDisc, sampleDim, agentLength);
    float* initial = new float[sampleDim];
    float* goal = new float[sampleDim];
    initial[0] = 5;
    initial[1] = 5;
    initial[2] = 0;
    initial[3] = 0;
    initial[4] = 0;
    initial[5] = 0;
    initial[6] = 0;
    goal[0] = 10;
    goal[1] = 10;
    goal[2] = 0;
    goal[3] = 0;
    goal[4] = 0;
    goal[5] = 0;
    goal[6] = 0;
    kgmt.plan(initial, goal);
    delete[] initial;
    delete[] goal;
    return 0;
}