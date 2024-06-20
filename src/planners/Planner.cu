#include "planners/Planner.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "agent/Agent.h"
#include "state/State.h"
#include <curand_kernel.h>

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

// Host function to manage GPU memory and launch kernel.
// One sample per thread.
// Each sample generates trajectory.
float* Planner::generateRandomSamples(const State &state, int numSamples) {
    
    float h_parent[4] = {state.x_, state.y_, state.theta_, state.v_}; // Parent state. Host memory
    float *d_parent; // Parent state. Device memory
    float* h_samples = new float[numSamples * 4]; // Samples. Host memory
    float *d_samples; // Samples. Device memory
    float h_controls[numSamples * 3]; // Controls. Host memory
    float *d_controls; // Controls. Device memory

    // Allocate memory on the device
    int size_parent = 4 * sizeof(float);
    int size_samples = numSamples * 4 * sizeof(float);
    int size_controls = numSamples * 3 * sizeof(float);
    checkCudaError(cudaMalloc((void **)&d_parent, size_parent), "Failed to allocate device memory for parent state");
    checkCudaError(cudaMalloc((void **)&d_samples, size_samples), "Failed to allocate device memory for samples");
    checkCudaError(cudaMalloc((void **)&d_controls, size_controls), "Failed to allocate device memory for controls");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_parent, h_parent, size_parent, cudaMemcpyHostToDevice), "Failed to copy parent state to device");

    // Sizing grid and block.
    dim3 dimGrid(1, 1, 1); // example. 1 block
    dim3 dimBlock(numSamples, 1, 1); // example. numSamples threads per block

    // call kernel
    generateRandomSampleKernel<<<dimGrid, dimBlock>>>(d_parent, d_samples, d_controls, numSamples, 1.0);

    // Copy data from device to host
    checkCudaError(cudaMemcpy(h_samples, d_samples, size_samples, cudaMemcpyDeviceToHost), "Failed to copy samples to host");
    checkCudaError(cudaMemcpy(h_controls, d_controls, size_controls, cudaMemcpyDeviceToHost), "Failed to copy controls to host");

    // Free memory on the device
    checkCudaError(cudaFree(d_parent), "Failed to free device memory for parent state");
    checkCudaError(cudaFree(d_samples), "Failed to free device memory for samples");

    return h_samples;

}