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
    int N = 100;
    int numIterations = 10;
    int maxSamples = 100000;
    int numDisc = 10;
    float agentLength = 1.0;

    KGMT kgmt(width, height, N, numIterations, maxSamples, numDisc, sampleDim, agentLength);
    float* initial = new float[sampleDim];
    float* goal = new float[sampleDim];
    initial[0] = 1;
    initial[1] = 1;
    initial[2] = 1;
    initial[3] = 1;
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