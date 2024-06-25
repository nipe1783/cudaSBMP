#include "planners/Planner.cuh"
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
#define length 1.0

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__
void propagateState(float* x0, float* controls, float* x1, int numDisc){
    printf("Block: %d, Thread: %d, paent: %f, %f, %f, %f\n", blockIdx.x, threadIdx.x, x0[0], x0[1], x0[2], x0[3]);
    float dt = controls[2] / numDisc;
    x1[0] = x0[0];
    x1[1] = x0[1];
    x1[2] = x0[2];
    x1[3] = x0[3];

    for (int i = 0; i < numDisc; i++) {
        x1[0] = x1[0] + x1[3] * cosf(x1[2]) * dt;
        x1[1] = x1[1] + x1[3] * sinf(x1[2]) * dt;
        x1[2] = x1[2] + x1[3] / length * tanf(controls[1]) * dt;
        x1[3] = x1[3] + controls[0] * dt;
    }
}

// Global: called from the CPU to the GPU
__global__ 
void generateRandomTreeKernel(const float* parents, float* tree, float* controls, const int numIterations) {
    
    // Load sample onto shared memory
    __shared__ float parent[STATE_DIM];
    if (threadIdx.x < STATE_DIM) {
        parent[threadIdx.x] = parents[threadIdx.x];
    }
    __syncthreads();
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int outRow = 0; outRow < numIterations; outRow++) {
        int outIndex = outRow * gridDim.x * blockDim.x * STATE_DIM + col * STATE_DIM;
        printf("Block: %d, Thread: %d, outIndex: %d\n", blockIdx.x, threadIdx.x, outIndex);
        int controlIndex = outRow * gridDim.x * blockDim.x * INPUT_DIM + col * INPUT_DIM;

        // Seed random number generator
        curandState state;
        curand_init((unsigned long long)clock() + outIndex, 0, 0, &state);

        // Generate random controls
        float a = curand_uniform(&state) * 5.0f - 2.5f;  // Scale to range [-2.5, 2.5]
        float steeringAngle = curand_uniform(&state) * M_PI - M_PI / 2;  // Scale to range [-pi/2, pi/2]
        float dt = curand_uniform(&state) * 0.3f;  // Scale to range [0, .3]

        // Write to controls array
        controls[controlIndex] = a;
        controls[controlIndex + 1] = steeringAngle;
        controls[controlIndex + 2] = dt;

        // Write to tree array
        float* x1 = &tree[outIndex];
        propagateState(parent, &controls[controlIndex], x1, 20);

        __syncthreads();

        // Update parent state for the next iteration
        if (threadIdx.x < STATE_DIM) {
            parent[threadIdx.x] = tree[outIndex + threadIdx.x];
        }
        __syncthreads();
    }
}

void Planner::generateRandomTree(const float* parent, const int numSamples, float **samples, float **controls){

    // initialize execution parameters
    const int threadsPerBlock = 4;
    const int blocksPerGrid = 2;
    const int rowsTree = 3;
    const int colsTree = STATE_DIM * threadsPerBlock * blocksPerGrid;
    const int colsControls = INPUT_DIM * threadsPerBlock * blocksPerGrid;
    int sizeTree = rowsTree * colsTree * sizeof(float);
    int sizeControls = rowsTree * colsControls * sizeof(float);
    int sizeParent = STATE_DIM * sizeof(float);
    dim3 dimGrid(blocksPerGrid, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);
    float milliseconds = 0;

    // host variables
    float hTree[rowsTree][colsTree];
    float hControls[rowsTree][colsControls];

    // device variables
    float *dTree;
    float *dControls;
    float *dParent;

    // allocate memory on device
    checkCudaError(cudaMalloc((void **)&dTree, sizeTree), "Failed to allocate device memory for tree");
    checkCudaError(cudaMalloc((void **)&dControls, sizeControls), "Failed to allocate device memory for controls");
    checkCudaError(cudaMalloc((void **)&dParent, sizeParent), "Failed to allocate device memory for parent state");
    
    // copy parent state to device
    checkCudaError(cudaMemcpy(dParent, parent, sizeParent, cudaMemcpyHostToDevice), "Failed to copy parent state to device");
    
    // Initialize Timer
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    checkCudaError(cudaEventRecord(start), "Failed to record start event");

    // Call the kernel
    generateRandomTreeKernel<<<dimGrid, dimBlock>>>(dParent, dTree, dControls, rowsTree);

    // Stop Timer
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to calculate elapsed time");

    // Transfer data from device to host
    checkCudaError(cudaMemcpy(hTree, dTree, sizeTree, cudaMemcpyDeviceToHost), "Failed to copy tree to host");
    checkCudaError(cudaMemcpy(hControls, dControls, sizeControls, cudaMemcpyDeviceToHost), "Failed to copy controls to host");

    // Free device memory
    checkCudaError(cudaFree(dTree), "Failed to free device memory for tree");
    checkCudaError(cudaFree(dControls), "Failed to free device memory for controls");
    checkCudaError(cudaFree(dParent), "Failed to free device memory for parent state");

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

    FILE *fp2 = fopen("controls.csv", "w");
    for (int i = 0; i < rowsTree; i++){
        for (int j = 0; j < colsControls; j++){
            fprintf(fp2, "%f,", hControls[i][j]);
        }
        fprintf(fp2, "\n");
    }
    fclose(fp2);
}