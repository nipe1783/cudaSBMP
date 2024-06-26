#include "planners/NaivePlanner.cuh"
#include "helper/helper.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "agent/Agent.h"
#include "state/State.h"
#include <curand_kernel.h>
#include <chrono>


#define STATE_DIM 4 // x,y,theta,v
#define INPUT_DIM 3 // a, steering angle, dt
#define SAMPLE_DIM 7 // x,y,theta,v,a,steering angle, dt  
#define length 1.0

void NaivePlanner::plan(float* start, float* goal){
    
    float* samples = nullptr;
    float* controls = nullptr;
    generateRandomTree(start, 100, &samples);
}

__device__
void propagateState(float* x0, float* x1, int numDisc, int seed){
    // Seed random number generator
    curandState state;
    curand_init(seed, 0, 0, &state);

    // Generate random controls
    x1[4] = curand_uniform(&state) * 5.0f - 2.5f;  // Scale to range [-2.5, 2.5]
    x1[5] = curand_uniform(&state) * M_PI - M_PI / 2;  // Scale to range [-pi/2, pi/2]
    x1[6] = curand_uniform(&state) * 0.3f;  // Scale to range [0, .3]

    float dt = x1[6] / numDisc;
    x1[0] = x0[0];
    x1[1] = x0[1];
    x1[2] = x0[2];
    x1[3] = x0[3];

    for (int i = 0; i < numDisc; i++) {
        x1[0] = x1[0] + x1[3] * cosf(x1[2]) * dt;
        x1[1] = x1[1] + x1[3] * sinf(x1[2]) * dt;
        x1[2] = x1[2] + x1[3] / length * tanf(x1[5]) * dt;
        x1[3] = x1[3] + x1[4] * dt;
    }
}


// Global: called from the CPU to the GPU
__global__ 
void generateRandomTreeKernel(const float* root, float* tree, const int numIterations, int tWidth) {
    
    // Load sample onto shared memory
    __shared__ float x0[STATE_DIM];
    if (threadIdx.x < STATE_DIM) {
        x0[threadIdx.x] = root[threadIdx.x];
    }
    __syncthreads();
    
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * SAMPLE_DIM;
    for (int row = 0; row < numIterations; row++) {
        int outIndex = row * tWidth + col;
        propagateState(x0, &tree[outIndex], 20, outIndex);

        __syncthreads();
        if (threadIdx.x < STATE_DIM) {
            int x0Col = blockIdx.x * blockDim.x * SAMPLE_DIM;
            int x0Index = row * tWidth + x0Col + threadIdx.x;
            x0[threadIdx.x] = root[x0Index];
        }
    }
}

void NaivePlanner::generateRandomTree(const float* root, const int numSamples, float **samples){
    // initialize execution parameters
    const int threadsPerBlock = 32;
    const int blocksPerGrid = 32;
    const int rowsTree = 10;
    const int colsTree = SAMPLE_DIM * threadsPerBlock * blocksPerGrid;
    const int tWidth = SAMPLE_DIM*threadsPerBlock*blocksPerGrid;
    int sizeTree = rowsTree * colsTree * sizeof(float);
    int sizeSample = SAMPLE_DIM * sizeof(float);
    dim3 dimGrid(blocksPerGrid, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);
    float milliseconds = 0;

    // host variables
    float hTree[rowsTree][colsTree];

    // device variables
    float *dTree;
    float *dRoot;

    // allocate memory on device
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dTree, sizeTree));
    CUDA_ERROR_CHECK(cudaMalloc((void **)&dRoot, sizeSample));
    
    // copy parent state to device
    CUDA_ERROR_CHECK(cudaMemcpy(dRoot, root, sizeSample, cudaMemcpyHostToDevice));
    
    // Initialize Timer
    cudaEvent_t start, stop;
    CUDA_ERROR_CHECK(cudaEventCreate(&start));
    CUDA_ERROR_CHECK(cudaEventCreate(&stop));
    CUDA_ERROR_CHECK(cudaEventRecord(start));

    // Call the kernel
    generateRandomTreeKernel<<<dimGrid, dimBlock>>>(dRoot, dTree, rowsTree, tWidth);

    // Stop Timer
    CUDA_ERROR_CHECK(cudaEventRecord(stop));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Transfer data from device to host
    CUDA_ERROR_CHECK(cudaMemcpy(hTree, dTree, sizeTree, cudaMemcpyDeviceToHost));

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
    FILE *fp = fopen("samples.csv", "w");
    for (int i = 0; i < rowsTree; i++){
        for (int j = 0; j < colsTree; j++){
            fprintf(fp, "%f,", hTree[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}