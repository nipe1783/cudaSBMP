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
    int sampleDim = 4;
    float width = 10.0;
    float height = 10.0;
    int N = 100;
    int numIterations = 1;
    int numSamples = 10000;
    int numDisc = 10;

    KGMT kgmt(width, height, N, numIterations, numSamples, numDisc, sampleDim);
    float* initial = new float[4];
    float* goal = new float[4];
    initial[0] = 1;
    initial[1] = 1;
    initial[2] = 1;
    initial[3] = 1;
    goal[0] = 10;
    goal[1] = 10;
    goal[2] = 0;
    goal[3] = 0;
    kgmt.plan(initial, goal);
    delete[] initial;
    delete[] goal;
    return 0;
}