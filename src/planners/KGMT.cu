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

KGMT::KGMT(float width, float height, int N, int numIterations, int maxSamples, int numDisc, int sampleDim, float agentLength):
    width_(width), height_(height), N_(N), numIterations_(numIterations), maxSamples_(maxSamples), numDisc_(numDisc), sampleDim_(sampleDim), agentLength_(agentLength){

    d_eOpen_ = thrust::device_vector<bool>(maxSamples);
    d_eClosed_ = thrust::device_vector<bool>(maxSamples);
    d_G_ = thrust::device_vector<bool>(maxSamples);
    d_eUnexplored_ = thrust::device_vector<bool>(maxSamples);
    d_edges_ = thrust::device_vector<int>(maxSamples);
    d_scanIdx_ = thrust::device_vector<int>(maxSamples);
    d_activeGIdx_ = thrust::device_vector<int>(maxSamples);
    d_activeIdx_ = thrust::device_vector<int>(maxSamples);
    d_samples_ = thrust::device_vector<float>(maxSamples * sampleDim);
    d_eConnectivity_ = thrust::device_vector<float>(maxSamples);

    d_eOpen_ptr_ = thrust::raw_pointer_cast(d_eOpen_.data());
    d_eUnexplored_ptr_ = thrust::raw_pointer_cast(d_eUnexplored_.data());
    d_eClosed_ptr_ = thrust::raw_pointer_cast(d_eClosed_.data());
    d_G_ptr_ = thrust::raw_pointer_cast(d_G_.data());
    d_samples_ptr_ = thrust::raw_pointer_cast(d_samples_.data());
    d_scanIdx_ptr_ = thrust::raw_pointer_cast(d_scanIdx_.data());
    d_activeIdx_G_ptr_ = thrust::raw_pointer_cast(d_activeGIdx_.data());
    d_activeIdx_ptr_ = thrust::raw_pointer_cast(d_activeIdx_.data());
    d_eConnectivity_ptr_ = thrust::raw_pointer_cast(d_eConnectivity_.data());

    cudaMalloc(&d_costGoal, sizeof(float));
}

void KGMT::plan(float* initial, float* goal) {
    double t_kgmtStart = std::clock();

    // initialize vectors with root of tree
    cudaMemcpy(d_samples_ptr_, initial, sampleDim_ * sizeof(float), cudaMemcpyHostToDevice);
    bool value = true;
    cudaMemcpy(d_G_ptr_, &value, sizeof(bool), cudaMemcpyHostToDevice);

    const int blockSize = 128;
    // const int gridSize = std::min((maxSamples_ + blockSize - 1) / blockSize, 2147483647);
    const int gridSize = 1;

    int itr = 0;
    treeSize_ = 1;
    connThresh_ = 0.0;
    int activeSize = 0;
    while(itr < numIterations_){
        itr++;

        // find total number of samples in G.
        thrust::exclusive_scan(d_G_.begin(), d_G_.end(), d_scanIdx_.begin());
        activeSize = d_scanIdx_[maxSamples_-1];
        (d_G_[maxSamples_ - 1]) ? ++activeSize : 0;
        printf("Iteration %d, active G Size %d\n", itr, activeSize);

        // find indices of active samples in G.
        findInd<<<gridSize, blockSize>>>(maxSamples_, d_G_ptr_, d_scanIdx_ptr_, d_activeIdx_G_ptr_);

        // expand active samples in G. Add new samples to eUnexplored.
        const int blockSizeActive = 128;
        // const int gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 2147483647);
        const int gridSizeActive = 1;
        propagateG<<<gridSizeActive, blockSizeActive>>>(d_samples_ptr_, d_eUnexplored_ptr_, d_eClosed_ptr_, d_G_ptr_, d_eConnectivity_ptr_, d_activeIdx_G_ptr_, activeSize, treeSize_, sampleDim_, agentLength_, numDisc_);
        
        // move samples from eUnexplored to eOpen.
        thrust::exclusive_scan(d_eUnexplored_.begin(), d_eUnexplored_.end(), d_scanIdx_.begin());
        activeSize = d_scanIdx_[maxSamples_-1];
        (d_eUnexplored_[maxSamples_ - 1]) ? ++activeSize : 0;
        printDeviceVector(d_eConnectivity_ptr_, 10);
        findInd<<<gridSize, blockSize>>>(maxSamples_, d_eUnexplored_ptr_, d_scanIdx_ptr_, d_activeIdx_ptr_);
        expandEOpen<<<gridSizeActive, blockSizeActive>>>(d_eUnexplored_ptr_, d_eClosed_ptr_, d_eOpen_ptr_, d_eConnectivity_ptr_, d_activeIdx_ptr_, activeSize, connThresh_);


        cudaMemcpy(&costGoal_, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
    }

    double t_kgmt = (std::clock() - t_kgmtStart) / (double) CLOCKS_PER_SEC;
    std::cout << "time inside KGMT is " << t_kgmt << std::endl;
}

__global__
void findInd(int numSamples, bool* S, int* scanIdx, int* activeS){
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numSamples)
        return;
    if (!S[node]) {
        return;
    }
    activeS[scanIdx[node]] = node;
}

// TODO: Possibly make x0 a shared memory variable and all threads in a block propagate the same sample.
// Can I make the memory access of samples coalesced? All G nodes are next to eachother in samples.
__global__
void propagateG(float* samples, bool* eUnexplored, bool* eClosed, bool* G, float* eConn, int* activeIdx_G, int activeSize_G, int treeSize, int sampleDim, float agentLength, int numDisc){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= activeSize_G)
        return;
    int x0Idx = activeIdx_G[tid];
    if(G[x0Idx]){
        float* x0 = &samples[x0Idx * sampleDim];
        float* x1 = &samples[treeSize * sampleDim];
        int x1Index = treeSize + tid;
        propagateState(x0, x1, numDisc, x1Index, agentLength);
        eConn[x1Index] = calculateConnectivity(x1);
        printf("X1 INDEX: %d, X1 CONN: %f\n", x1Index, eConn[x1Index]);
        eUnexplored[x1Index] = true;
        eClosed[x0Idx] = true; // TODO: Do I need to do this? Possibly remove eClosed and just use G.
        G[x0Idx] = false;
    }
}

__global__
void expandEOpen(bool* eUnexplored, bool* eClosed, bool* eOpen, float* eConn, int* activeEUnexplored_Idx, int size_activeEUnexplored, float connThresh){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size_activeEUnexplored)
        return;
    int xIDx = activeEUnexplored_Idx[tid];
    printf("Sample %d has connectivity %d\n", xIDx, eConn[xIDx]);
    if(eUnexplored[xIDx]){
        if(eConn[xIDx] > connThresh){
            printf("Sample %d has connectivity %d\n", xIDx, eConn[xIDx]);
            eOpen[xIDx] = true;
            eUnexplored[xIDx] = false;
            return;
        }
        eUnexplored[xIDx] = false;
        eClosed[xIDx] = true;
    }
}

__device__
void propagateState(float* x0, float* x1, int numDisc, int x1Index, float agentLength){
    // Seed random number generator
    curandState state;
    curand_init(x1Index, 0, 0, &state);

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
        x1[2] = x1[2] + x1[3] / agentLength * tanf(x1[5]) * dt;
        x1[3] = x1[3] + x1[4] * dt;
    }
}

// TODO: Implement this function.
// input is a sample x, output is the connectivity of the sample.
// determines a worth heuristic for the sample based on coverage.
__device__ float calculateConnectivity(float* x){
    return 20.0;
}
