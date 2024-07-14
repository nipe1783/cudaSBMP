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
#include "planners/KGMT.cuh"
#include "planners/GoalBiasedKGMT.cuh"
#include "helper/helper.cuh"
#include "config/config.h"

#define WORKSPACE_DIM 2

int main(void)
{
    int sampleDim       = 7;
    float width         = 20.0;
    float height        = 20.0;
    int N               = 16;
    int n               = 8;
    int numIterations   = 100;
    int maxTreeSize     = 30000;
    int numDisc         = 10;
    float agentLength   = 1.0;
    float goalThreshold = 0.5;

    GoalBiasedKGMT kgmtGB(WS_SIZE, WS_SIZE, R1, R2, MAX_ITER, MAX_TREE_SIZE, NUM_DISC, UNI_LENGTH, GOAL_THRESH);
    // KGMT kgmt(width, height, N, n, numIterations, maxTreeSize, numDisc, agentLength, goalThreshold);
    float h_initial[SAMPLE_DIM] = {.5, .5, 0.0, 0.0, 0.0, 0.0, 0.0}, h_goal[SAMPLE_DIM] = {10, 10, 0.0, 0.0, 0.0, 0.0, 0.0};

    int numObstacles = 2;
    float* d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV("../include/config/obstacles/obstacles.csv", numObstacles, WORKSPACE_DIM);
    CUDA_ERROR_CHECK(cudaMalloc(&d_obstacles, sizeof(float) * 2 * numObstacles * WORKSPACE_DIM));
    CUDA_ERROR_CHECK(cudaMemcpy(d_obstacles, obstacles.data(), sizeof(float) * 2 * numObstacles * WORKSPACE_DIM, cudaMemcpyHostToDevice));
    kgmtGB.plan(h_initial, h_goal, d_obstacles, numObstacles);
    // kgmt.plan(initial, goal, d_obstacles, numObstacles);

    cudaFree(d_obstacles);
    return 0;
}