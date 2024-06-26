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

int main(void){

    // Create a planner
    // CostPropPlanner rrt;
    // float* start = new float[4];
    // float* goal = new float[4];
    // start[0] = 0;
    // start[1] = 0;
    // start[2] = 0;
    // start[3] = 0;
    // goal[0] = 10;
    // goal[1] = 10;
    // goal[2] = 0;
    // goal[3] = 0;
    // rrt.plan(start, goal);
    // const OccupancyGrid grid(10.0, 10.0, 10);
    // float x = 0.0f;
    // float y = 0.0f;
    // int index = grid.getCellIndex(x, y);
    // std::cout << "Cell index: " << index << std::endl;

    int sampleDim = 7;
    float width = 10.0;
    float height = 10.0;
    int N = 100;
    int numIterations = 100;
    int numSamples = 10000;
    int numDisc = 10;



    KGMT kgmt(width, height, N, numIterations, numSamples, numDisc, sampleDim);
    float* initial = new float[4];
    float* goal = new float[4];
    initial[0] = 0;
    initial[1] = 0;
    initial[2] = 0;
    initial[3] = 0;
    goal[0] = 10;
    goal[1] = 10;
    goal[2] = 0;
    goal[3] = 0;
    kgmt.plan(initial, goal);
    return 0;
}