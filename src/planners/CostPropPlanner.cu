#include "planners/CostPropPlanner.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "agent/Agent.h"
#include "state/State.h"
#include "helper/helper.cuh"
#include <curand_kernel.h>
#include <chrono>


#define STATE_DIM 4 // x,y,theta,v
#define INPUT_DIM 3 // a, steering angle, dt
#define SAMPLE_DIM 7 // x,y,theta,v,a,steering angle, dt  
#define length 1.0

void CostPropPlanner::plan(float* start, float* goal){
    
    float* samples = nullptr;
    float* controls = nullptr;
    generateRandomTree(start, 100, &samples);
}

__device__
void propagateState(float* x0, float* x1, int numDisc, curandState* state) {
    // Generate random controls
    float a = curand_uniform(state) * 5.0f - 2.5f;  // Scale to range [-2.5, 2.5]
    float steering = curand_uniform(state) * M_PI - M_PI / 2;  // Scale to range [-pi/2, pi/2]
    float duration = curand_uniform(state) * 0.3f;  // Scale to range [0, .3]

    float dt = duration / numDisc;
    float x = x0[0];
    float y = x0[1];
    float theta = x0[2];
    float v = x0[3];

    float cos_theta, sin_theta, tan_steering;

    for (int i = 0; i < numDisc; i++) {
        cos_theta = cosf(theta);
        sin_theta = sinf(theta);
        tan_steering = tanf(steering);

        x += v * cos_theta * dt;
        y += v * sin_theta * dt;
        theta += (v / length) * tan_steering * dt;
        v += a * dt;
    }

    x1[0] = x;
    x1[1] = y;
    x1[2] = theta;
    x1[3] = v;
    x1[4] = a;
    x1[5] = steering;
    x1[6] = duration;
}

__global__
void generateRandomTreeKernel(const float* root, float* tree, const int numIterations, int tWidth) {
    __shared__ float x0[STATE_DIM];
    if (threadIdx.x < STATE_DIM) {
        x0[threadIdx.x] = root[threadIdx.x];
    }
    __syncthreads();

    curandState state;
    int seed = (blockIdx.x * blockDim.x + threadIdx.x) * numIterations;
    curand_init(seed, 0, 0, &state);
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * SAMPLE_DIM;
    for (int row = 0; row < numIterations; row++) {
        int outIndex = row * tWidth + col;
        propagateState(x0, &tree[outIndex], 20, &state);
        if (threadIdx.x < STATE_DIM) {
            int x0Col = blockIdx.x * blockDim.x * SAMPLE_DIM;
            int x0Index = row * tWidth + x0Col + threadIdx.x;
            x0[threadIdx.x] = tree[x0Index];
        }
    }
}

void CostPropPlanner::generateRandomTree(const float* root, const int numSamples, float **samples){
    // initialize execution parameters
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = 512;
    const int rowsTree = 1;
    const int colsTree = SAMPLE_DIM * threadsPerBlock * blocksPerGrid;
    int sizeTree = rowsTree * colsTree * sizeof(float);
    int sizeSample = SAMPLE_DIM * sizeof(float);
    dim3 dimGrid(blocksPerGrid, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);
    float milliseconds = 0;

    // host variables
    float* hTree = new float[rowsTree * colsTree];

    // device variables
    float *dTree;
    float *dRoot;

    // allocate memory on device
    CUDA_ERROR_CHECK( cudaMalloc((void **)&dTree, sizeTree));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dRoot, sizeSample));
    
    // copy parent state to device
    CUDA_ERROR_CHECK(cudaMemcpy(dRoot, root, sizeSample, cudaMemcpyHostToDevice));
    
    // Initialize Timer
    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));
    CUDA_ERROR_CHECK(cudaEventRecord(start));

    // Call the kernel
    generateRandomTreeKernel<<<dimGrid, dimBlock>>>(dRoot, dTree, rowsTree, colsTree);

    // Transfer data from device to host
    CUDA_ERROR_CHECK(cudaMemcpy(hTree, dTree, sizeTree, cudaMemcpyDeviceToHost));

    // Stop Timer
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Free device memory
    CUDA_ERROR_CHECK(cudaFree(dTree));
    CUDA_ERROR_CHECK(cudaFree(dRoot));

    // Destroy CUDA events
    CUDA_ERROR_CHECK(cudaEventDestroy(start));
    CUDA_ERROR_CHECK(cudaEventDestroy(stop));

    // Print Information
    printf("Kernel execution time: %f milliseconds\n", milliseconds);
    printf("Tree size: %d\n", rowsTree*colsTree);

    // Save data to CSV
    // FILE *fp = fopen("samples.csv", "w");
    // for (int i = 0; i < rowsTree; i++){
    //     for (int j = 0; j < colsTree; j++){
    //         int ind = i * colsTree + j;
    //         fprintf(fp, "%f,", hTree[ind]);
    //     }
    //     fprintf(fp, "\n");
    // }
    // fclose(fp);
    delete[] hTree;
}