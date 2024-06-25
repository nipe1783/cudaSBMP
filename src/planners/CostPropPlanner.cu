#include "planners/CostPropPlanner.cuh"
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

void CostPropPlanner::plan(float* start, float* goal){
    
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

void CostPropPlanner::generateRandomTree(const float* root, const int numSamples, float **samples){
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
    checkCudaError(cudaMalloc((void **)&dTree, sizeTree), "Failed to allocate device memory for tree");
    checkCudaError(cudaMalloc((void **)&dRoot, sizeSample), "Failed to allocate device memory for parent state");
    
    // copy parent state to device
    checkCudaError(cudaMemcpy(dRoot, root, sizeSample, cudaMemcpyHostToDevice), "Failed to copy parent state to device");
    
    // Initialize Timer
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    checkCudaError(cudaEventRecord(start), "Failed to record start event");

    // Call the kernel
    generateRandomTreeKernel<<<dimGrid, dimBlock>>>(dRoot, dTree, rowsTree, tWidth);

    // Stop Timer
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to calculate elapsed time");

    // Transfer data from device to host
    checkCudaError(cudaMemcpy(hTree, dTree, sizeTree, cudaMemcpyDeviceToHost), "Failed to copy tree to host");

    // Free device memory
    checkCudaError(cudaFree(dTree), "Failed to free device memory for tree");
    checkCudaError(cudaFree(dRoot), "Failed to free device memory for parent state");

    // Destroy CUDA events
    checkCudaError(cudaEventDestroy(start), "Failed to destroy start event");
    checkCudaError(cudaEventDestroy(stop), "Failed to destroy stop event");

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