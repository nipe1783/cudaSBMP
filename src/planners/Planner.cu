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

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


// Global: called from the CPU to the GPU
__global__ void generateRandomSampleKernel(const float *parent, float *samples, float *controls, const int numSamples, const float length){
    // sample: [x, y, theta, v]
    int sampleIdx = threadIdx.x;

    curandState state;
    curand_init((unsigned long long)clock() + sampleIdx, 0, 0, &state);

    // Generate random controls
    float a = curand_uniform(&state);
    float steeringAngle = curand_uniform(&state);
    float dt = curand_uniform(&state);

    controls[sampleIdx * 3] = a;
    controls[sampleIdx * 3 + 1] = steeringAngle;
    controls[sampleIdx * 3 + 2] = dt;

    // Generate sample
    samples[sampleIdx * 4] = parent[0] + parent[3] * cos(parent[2]) * dt;
    samples[sampleIdx * 4 + 1] = parent[1] + parent[3] * sin(parent[2]) * dt;
    samples[sampleIdx * 4 + 2] = parent[2] + parent[3]/length * tan(steeringAngle) * dt;
    samples[sampleIdx * 4 + 3] = parent[3] + a * dt;
}

// Global: called from the CPU to the GPU
__global__ void generateRandomSampleKernelV2(const float *parents, float *samples, float *controls, const int numSamples, const float length){
    int sampleIdx = blockIdx.x * STATE_DIM + threadIdx.x;

    // load sample onto shared memory
    __shared__ float parent[STATE_DIM];
    if(threadIdx.x < STATE_DIM){
        parent[threadIdx.x] = parents[blockIdx.x * STATE_DIM + threadIdx.x];
    }
    __syncthreads();

    // seed random number generator
    curandState state;
    curand_init((unsigned long long)clock() + sampleIdx, 0, 0, &state);

    // Generate random controls
    float a = curand_uniform(&state);
    float steeringAngle = curand_uniform(&state);
    float dt = curand_uniform(&state);
    controls[sampleIdx * 3] = a;
    controls[sampleIdx * 3 + 1] = steeringAngle;
    controls[sampleIdx * 3 + 2] = dt;

    // Generate sample
    samples[sampleIdx * 4] = parent[0] + parent[3] * cos(parent[2]) * dt;
    samples[sampleIdx * 4 + 1] = parent[1] + parent[3] * sin(parent[2]) * dt;
    samples[sampleIdx * 4 + 2] = parent[2] + parent[3]/length * tan(steeringAngle) * dt;
    samples[sampleIdx * 4 + 3] = parent[3] + a * dt;

}

// Host function to manage GPU memory and launch kernel.
// One sample per thread.
// Each sample generates trajectory.
void Planner::generateRandomSamples(const float* parent, const int numSamples, float **samples, float **controls) {
    
    float *d_parent; // Parent state. Device memory
    float *h_samples = new float[numSamples * 4]; // Samples. Host memory
    float *d_samples; // Samples. Device memory
    float *h_controls = new float[numSamples * 3]; // Controls. Host memory
    float *d_controls; // Controls. Device memory

    // Allocate memory on the device
    int size_parent = 4 * sizeof(float);
    int size_samples = numSamples * 4 * sizeof(float);
    int size_controls = numSamples * 3 * sizeof(float);
    checkCudaError(cudaMalloc((void **)&d_parent, size_parent), "Failed to allocate device memory for parent state");
    checkCudaError(cudaMalloc((void **)&d_samples, size_samples), "Failed to allocate device memory for samples");
    checkCudaError(cudaMalloc((void **)&d_controls, size_controls), "Failed to allocate device memory for controls");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_parent, parent, size_parent, cudaMemcpyHostToDevice), "Failed to copy parent state to device");

    // Sizing grid and block.
    dim3 dimGrid(1, 1, 1); // example. 1 block
    dim3 dimBlock(numSamples, 1, 1); // example. numSamples threads per block

    // call kernel
    generateRandomSampleKernelV2<<<dimGrid, dimBlock>>>(d_parent, d_samples, d_controls, numSamples, 1.0);

    // Copy data from device to host
    checkCudaError(cudaMemcpy(h_samples, d_samples, size_samples, cudaMemcpyDeviceToHost), "Failed to copy samples to host");
    checkCudaError(cudaMemcpy(h_controls, d_controls, size_controls, cudaMemcpyDeviceToHost), "Failed to copy controls to host");

    // Free memory on the device
    checkCudaError(cudaFree(d_parent), "Failed to free device memory for parent state");
    checkCudaError(cudaFree(d_samples), "Failed to free device memory for samples");

    // Assign pointers to the outputs
    *samples = h_samples;
    *controls = h_controls;

}

// Global: called from the CPU to the GPU
__global__ void generateRandomTreeKernel(const float *parents, float *tree, float *controls, const int numIterations, const float length){
    
    // load sample onto shared memory
    __shared__ float parent[STATE_DIM];
    if(threadIdx.x < STATE_DIM){
        parent[threadIdx.x] = parents[threadIdx.x];
    }
    __syncthreads();
    
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    for (int outRow = 0; outRow < numIterations; outRow++){
        int outIndex = outRow*STATE_DIM + (outRow * gridDim.x * blockDim.x + col) * STATE_DIM;
        // seed random number generator
        curandState state;
        curand_init((unsigned long long)clock() + outIndex, 0, 0, &state);
        // Generate random controls
        float a = curand_uniform(&state);
        float steeringAngle = curand_uniform(&state);
        float dt = curand_uniform(&state);
        tree[outIndex] = parent[0] + parent[3] * cos(parent[2]) * dt;
        tree[outIndex + 1] = parent[1] + parent[3] * sin(parent[2]) * dt;
        tree[outIndex + 2] = parent[2] + parent[3]/length * tan(steeringAngle) * dt;
        tree[outIndex + 3] = parent[3] + a * dt;
        if(threadIdx.x < STATE_DIM){
            printf("New Parent Idx: %d\n", outRow*STATE_DIM + threadIdx.x);
            parent[threadIdx.x] = tree[outRow*STATE_DIM + threadIdx.x];
        }
        __syncthreads();
    }
}

void Planner::generateRandomTree(const float* parent, const int numSamples, float **samples, float **controls){

    const int threadsPerBlock = 5;
    const int blocksPerGrid = 1;
    const int rowsTree = 3;
    const int colsTree = STATE_DIM * threadsPerBlock * blocksPerGrid;

    float hTree[rowsTree][colsTree];
    int sizeTree = rowsTree * colsTree * sizeof(float);

    float *dTree;
    checkCudaError(cudaMalloc((void **)&dTree, sizeTree), "Failed to allocate device memory for tree");

    float *dParent;
    checkCudaError(cudaMalloc((void **)&dParent, STATE_DIM * sizeof(float)), "Failed to allocate device memory for parent state");
    checkCudaError(cudaMemcpy(dParent, parent, STATE_DIM * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy parent state to device");

     // Create CUDA events
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");

    // Record the start event
    checkCudaError(cudaEventRecord(start), "Failed to record start event");

    // Set kernel parameters and call kernel
    dim3 dimGrid(blocksPerGrid, 1, 1);
    dim3 dimBlock(threadsPerBlock, 1, 1);
    generateRandomTreeKernel<<<dimGrid, dimBlock>>>(dParent, dTree, *controls, rowsTree, 1.0);

    // Record the stop event
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");

    // Wait for the stop event to complete
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");

    // Calculate the elapsed time
    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to calculate elapsed time");

    // Copy data from device to host
    checkCudaError(cudaMemcpy(hTree, dTree, sizeTree, cudaMemcpyDeviceToHost), "Failed to copy tree to host");

    // Free device memory
    checkCudaError(cudaFree(dTree), "Failed to free device memory for tree");

    // Destroy CUDA events
    checkCudaError(cudaEventDestroy(start), "Failed to destroy start event");
    checkCudaError(cudaEventDestroy(stop), "Failed to destroy stop event");

    // Print the elapsed time
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    // print tree size:
    printf("Tree size: %d\n", rowsTree*colsTree);

    // print tree:
    // for (int i = 0; i < rowsTree; ++i) {
    //     for (int j = 0; j < colsTree; ++j) {
    //         printf("%f ", hTree[i][j]);
    //     }
    //     printf("\n");
    // }

    // timing CPU version:
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < rowsTree; ++i) {
        for (int j = 0; j < colsTree; ++j) {
            hTree[i][j] = 0.0f; // Fill with zeros for example
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

    // Print CPU execution time
    printf("CPU execution time: %f milliseconds\n", cpu_duration.count());





}