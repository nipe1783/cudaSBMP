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
#include "helper/helper.cuh"

#define WORKSPACE_DIM 2

int main(void) {
    int sampleDim = 7;
    float width = 20.0;
    float height = 20.0;
    int N = 16;
    int n = 8;
    int numIterations = 100;
    int maxTreeSize = 30000;
    int numDisc = 10;
    float agentLength = 1.0;
    float goalThreshold = 0.5;

    KGMT kgmt(width, height, N, n, numIterations, maxTreeSize, numDisc, agentLength, goalThreshold);
    float* initial = new float[sampleDim];
    float* goal = new float[sampleDim];
    initial[0] = 5;
    initial[1] = 5;
    initial[2] = 0;
    initial[3] = 0;
    initial[4] = 0;
    initial[5] = 0;
    initial[6] = 0;
    goal[0] = 2;
    goal[1] = 18;
    goal[2] = 0;
    goal[3] = 0;
    goal[4] = 0;
    goal[5] = 0;
    goal[6] = 0;

    
    int numObstacles = 2;
    float *d_obstacles;
    std::vector<float> obstacles = readObstaclesFromCSV("../configurations/obstacles/obstacles.csv", numObstacles, WORKSPACE_DIM);
    printf("Obstacles: \n");
    for (int i = 0; i < numObstacles; i++) {
        for (int j = 0; j < 2 * WORKSPACE_DIM; j++) {
            printf("%f ", obstacles[i * 2 * WORKSPACE_DIM + j]);
        }
        printf("\n");
    }
    printf("numObstacles: %d\n", numObstacles);
	CUDA_ERROR_CHECK(cudaMalloc(&d_obstacles, sizeof(float)*2*numObstacles*WORKSPACE_DIM));
	CUDA_ERROR_CHECK(cudaMemcpy(d_obstacles, obstacles.data(), sizeof(float)*2*numObstacles*WORKSPACE_DIM, cudaMemcpyHostToDevice));
    kgmt.plan(initial, goal, d_obstacles, numObstacles);

    cudaFree(d_obstacles);
    delete[] initial;
    delete[] goal;
    return 0;
}