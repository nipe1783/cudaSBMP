#include "planners/KGMT.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "agent/Agent.h"
#include "state/State.h"
#include "helper/helper.cuh"
#include "helper/helper.cu"
#include <curand_kernel.h>
#include <chrono>

KGMT::KGMT(float width, float height, int N, int maxSamples, int numSamples, int numDisc, int sampleDim)
    : width_(width), height_(height), N_(N), sampleDim_(sampleDim), maxSamples_(maxSamples), numSamples_(numSamples), numDisc_(numDisc),
      cellSize_(width / N), grid_(width, height, N) {

    d_eOpen_ = thrust::device_vector<bool>(numSamples);
    d_eClosed_ = thrust::device_vector<bool>(numSamples);
    d_G_ = thrust::device_vector<bool>(numSamples);
    d_edges_ = thrust::device_vector<int>(numSamples);
    d_samples_ = thrust::device_vector<float>(numSamples * sampleDim);
    d_scanIdx_ = thrust::device_vector<int>(numSamples);
    d_activeGIdx_ = thrust::device_vector<int>(numSamples);

    d_eOpen_ptr_ = thrust::raw_pointer_cast(d_eOpen_.data());
    d_eClosed_ptr_ = thrust::raw_pointer_cast(d_eClosed_.data());
    d_G_ptr_ = thrust::raw_pointer_cast(d_G_.data());
    d_samples_ptr_ = thrust::raw_pointer_cast(d_samples_.data());
    d_scanIdx_ptr_ = thrust::raw_pointer_cast(d_scanIdx_.data());
    d_activeGIdx_ptr_ = thrust::raw_pointer_cast(d_activeGIdx_.data());

    cudaMalloc(&d_costGoal, sizeof(float));
}

void KGMT::plan(float* initial, float* goal) {
    double t_kgmtStart = std::clock();

    // initialize vectors with root of tree
    cudaMemcpy(d_samples_ptr_, initial, sampleDim_ * sizeof(float), cudaMemcpyHostToDevice);
    bool value = true;
    cudaMemcpy(d_G_ptr_, &value, sizeof(bool), cudaMemcpyHostToDevice);

    const int blockSize = 128;
	const int gridSize = std::min((maxSamples_ + blockSize - 1) / blockSize, 2147483647);
    // printf("gridSize is %d\n", gridSize);

    int itr = 0;
    int activeSize = 0;
    while(itr < numIterations_){
        itr++;

        // find total number of samples in G.
        thrust::exclusive_scan(d_G_.begin(), d_G_.end(), d_scanIdx_.begin());
        activeSize = d_scanIdx_[maxSamples_-1];
        (d_G_[d_G_.size() - 1]) ? ++activeSize : 0;

        // find indices of active samples in G.
        fillG<<<gridSize, blockSize>>>(maxSamples_, d_G_ptr_, d_scanIdx_ptr_, d_activeGIdx_ptr_);

        // expand active samples in G.
        const int blockSizeActive = 128;
        const int gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 2147483647);
        expandG<<<gridSizeActive, blockSizeActive>>>(d_samples_ptr_, d_activeGIdx_ptr_, activeSize);
        
        cudaMemcpy(&costGoal, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
    }

    double t_kgmt = (std::clock() - t_kgmtStart) / (double) CLOCKS_PER_SEC;
    std::cout << "time inside KGMT is " << t_kgmt << std::endl;
}

__global__
void fillG(int numSamples, bool* G, int* scanIdx, int* activeG){
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numSamples)
        return;
    if (!G[node]) {
        return;
    }
    activeG[scanIdx[node]] = node;
}

__global__
void expandG(float* samples, int* activeG, int activeSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= activeSize)
        return;
    // printf("activeG[tid] is %d\n", activeG[tid]);
}
