#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "planners/Planner.cuh"
#include "planners/RRT.cuh"

int main(void){

    // Create a planner
    RRT rrt;
    float* start = new float[4];
    float* goal = new float[4];
    start[0] = 0;
    start[1] = 0;
    start[2] = 0;
    start[3] = 0;
    goal[0] = 10;
    goal[1] = 10;
    goal[2] = 0;
    goal[3] = 0;
    rrt.plan(start, goal);
    return 0;
}