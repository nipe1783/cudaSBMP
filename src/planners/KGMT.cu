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
#include <ctime>

#define SAMPLE_DIM 7
#define STATE_DIM 4
#define BLOCK_SIZE 32

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
    d_eParentIdx_ = thrust::device_vector<int>(maxSamples);
    d_samples_ = thrust::device_vector<float>(maxSamples * sampleDim);
    d_eConnectivity_ = thrust::device_vector<float>(maxSamples);
    d_xGoal_ = thrust::device_vector<float>(sampleDim);

    d_eOpen_ptr_ = thrust::raw_pointer_cast(d_eOpen_.data());
    d_eUnexplored_ptr_ = thrust::raw_pointer_cast(d_eUnexplored_.data());
    d_eClosed_ptr_ = thrust::raw_pointer_cast(d_eClosed_.data());
    d_G_ptr_ = thrust::raw_pointer_cast(d_G_.data());
    d_samples_ptr_ = thrust::raw_pointer_cast(d_samples_.data());
    d_scanIdx_ptr_ = thrust::raw_pointer_cast(d_scanIdx_.data());
    d_activeIdx_G_ptr_ = thrust::raw_pointer_cast(d_activeGIdx_.data());
    d_activeIdx_ptr_ = thrust::raw_pointer_cast(d_activeIdx_.data());
    d_eConnectivity_ptr_ = thrust::raw_pointer_cast(d_eConnectivity_.data());
    d_eParentIdx_ptr_ = thrust::raw_pointer_cast(d_eParentIdx_.data());
    d_xGoal_ptr_ = thrust::raw_pointer_cast(d_xGoal_.data());

    cudaMalloc(&d_costGoal, sizeof(float));
    thrust::fill(d_eParentIdx_.begin(), d_eParentIdx_.end(), -1);
}

void KGMT::plan(float* initial, float* goal) {
    
    double t_kgmtStart = std::clock();

    // initialize vectors with root of tree
    cudaMemcpy(d_samples_ptr_, initial, sampleDim_ * sizeof(float), cudaMemcpyHostToDevice);
    bool value = true;
    cudaMemcpy(d_G_ptr_, &value, sizeof(bool), cudaMemcpyHostToDevice);

    // initialize xGoal
    cudaMemcpy(d_xGoal_ptr_, goal, sampleDim_ * sizeof(float), cudaMemcpyHostToDevice);
    
    const int blockSize = 128;
	const int gridSize = std::min((maxSamples_ + blockSize - 1) / blockSize, 2147483647);
    int gridSizeActive = 1;
    const int blockSizeActive = 32;

    // initialize random seed for curand
    curandState* d_states;
    cudaMalloc(&d_states, maxSamples_ * sizeof(curandState));
    initCurandStates<<<(maxSamples_ + blockSize - 1) / blockSize, blockSize>>>(d_states, maxSamples_, time(NULL));

    int itr = 0;
    treeSize_ = 1;
    connThresh_ = 0.0;
    int activeSize = 0;

    while(itr < numIterations_){
        itr++;

        // find total number of samples in G.
        thrust::exclusive_scan(d_G_.begin(), d_G_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxSamples_-1];
        (d_G_[maxSamples_ - 1]) ? ++activeSize : 0;

        // find indices of active samples in G.
        findInd<<<gridSize, blockSize>>>(maxSamples_, d_G_ptr_, d_scanIdx_ptr_, d_activeIdx_G_ptr_);

        // expand active samples in G. Add new samples to eUnexplored.
        // gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 16);
        gridSizeActive = 512;
        propagateG<<<gridSizeActive, blockSizeActive>>>(d_xGoal_ptr_, d_samples_ptr_, d_eUnexplored_ptr_, d_eClosed_ptr_, d_G_ptr_, d_eConnectivity_ptr_, d_eParentIdx_ptr_, d_activeIdx_G_ptr_, activeSize, treeSize_, sampleDim_, agentLength_, numDisc_, d_states);
        
        // move samples from eUnexplored to eOpen.
        thrust::exclusive_scan(d_eUnexplored_.begin(), d_eUnexplored_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxSamples_-1];
        (d_eUnexplored_[maxSamples_ - 1]) ? ++activeSize : 0;
        findInd<<<gridSize, blockSize>>>(maxSamples_, d_eUnexplored_ptr_, d_scanIdx_ptr_, d_activeIdx_ptr_);
        gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 2147483647);
        expandEOpen<<<gridSizeActive, blockSizeActive>>>(d_eUnexplored_ptr_, d_eClosed_ptr_, d_eOpen_ptr_, d_eConnectivity_ptr_, d_activeIdx_ptr_, activeSize, connThresh_);

        // move samples from eOpen to G.
        thrust::exclusive_scan(d_eOpen_.begin(), d_eOpen_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxSamples_-1];
        (d_eOpen_[maxSamples_ - 1]) ? ++activeSize : 0;
        findInd<<<gridSize, blockSize>>>(maxSamples_, d_eOpen_ptr_, d_scanIdx_ptr_, d_activeIdx_ptr_);
        gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 2147483647);
        expandG<<<gridSizeActive, blockSizeActive>>>(d_eOpen_ptr_, d_G_ptr_, d_eConnectivity_ptr_, d_activeIdx_ptr_, activeSize, connThresh_);

        treeSize_ += gridSizeActive * blockSizeActive;
        // treeSize_ += 1;
        cudaMemcpy(&costGoal_, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
    }

    double t_kgmt = (std::clock() - t_kgmtStart) / (double) CLOCKS_PER_SEC;
    std::cout << "time inside KGMT is " << t_kgmt << std::endl;

    // move vectors to csv to be plotted.
    copyAndWriteVectorToCSV(d_samples_, "samples.csv", maxSamples_, sampleDim_);
    copyAndWriteVectorToCSV(d_eParentIdx_, "parentRelations.csv", maxSamples_, 1);
    copyAndWriteVectorToCSV(d_eUnexplored_, "unexplored.csv", maxSamples_, 1);
    copyAndWriteVectorToCSV(d_eOpen_, "open.csv", maxSamples_, 1);
    copyAndWriteVectorToCSV(d_eClosed_, "closed.csv", maxSamples_, 1);
    copyAndWriteVectorToCSV(d_G_, "G.csv", maxSamples_, 1);
    copyAndWriteVectorToCSV(d_eConnectivity_, "connectivity.csv", maxSamples_, 1);

    // Free the allocated memory for curand states
    cudaFree(d_states);
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
// Extends one sample per sample in G.
// __global__
// void propagateG(float* samples, bool* eUnexplored, bool* eClosed, bool* G, float* eConn, int* eParentIDx, int* activeIdx_G, int activeSize_G, int treeSize, int sampleDim, float agentLength, int numDisc, curandState* states) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= activeSize_G)
//         return;
//     int x0Idx = activeIdx_G[tid];
//     if(G[x0Idx]){
//         float* x0 = &samples[x0Idx * sampleDim];
//         float* x1 = &samples[treeSize * sampleDim];
//         int x1Index = treeSize + tid;
//         curandState state = states[tid];
//         propagateState(x0, x1, numDisc, x1Index, agentLength, &state);
//         eConn[x1Index] = calculateConnectivity(x1);
//         eUnexplored[x1Index] = true;
//         eClosed[x0Idx] = true;  // TODO: Do I need to do this? Possibly remove eClosed and just use G.
//         G[x0Idx] = false;
//         eParentIDx[x1Index] = x0Idx;
//         states[tid] = state;
//     }
// }

// Extends blockDim.x samples per sample in G.
__global__
void propagateG(float* xGoal, float* samples, bool* eUnexplored, bool* eClosed, bool* G, float* eConn, int* eParentIDx, int* activeIdx_G, int activeSize_G, int treeSize, int sampleDim, float agentLength, int numDisc, curandState* states) {
    
    if (blockIdx.x >= activeSize_G)
        return;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    __shared__ int x0Idx;
    if (threadIdx.x == 0) {
        x0Idx = activeIdx_G[blockIdx.x];
    }
    __syncthreads();
    if (G[x0Idx]) {
        __shared__ float x0[SAMPLE_DIM];
        __shared__ float s_xGoal[SAMPLE_DIM];
        if (threadIdx.x < sampleDim) {
            x0[threadIdx.x] = samples[x0Idx * sampleDim + threadIdx.x];
            s_xGoal[threadIdx.x] = xGoal[threadIdx.x];
        }
        __syncthreads();
        float* x1 = &samples[treeSize * sampleDim + tid * sampleDim];
        int x1Index = treeSize + tid;

        curandState state = states[tid];
        propagateState(x0, x1, numDisc, agentLength, &state);
        eConn[x1Index] = calculateConnectivity(x1, s_xGoal);
        eUnexplored[x1Index] = true;
        eParentIDx[x1Index] = x0Idx;
        states[tid] = state;
        if (threadIdx.x == 0) {
            eClosed[x0Idx] = true;  // TODO: Possibly remove eClosed and just use G.
            G[x0Idx] = false;
        }
    }
}

// Extends blockDim.x samples per sample in G. Chooses maximum value. Uses parallel reduction.
// __global__ void propagateG(float* xGoal, float* samples, bool* eUnexplored, bool* eClosed, bool* G, float* eConn, int* eParentIDx, int* activeIdx_G, int activeSize_G, int treeSize, int sampleDim, float agentLength, int numDisc, curandState* states) {
//     if (blockIdx.x >= activeSize_G)
//         return;

//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     __shared__ int x0Idx;
//     __shared__ int x1Index;
//     if (threadIdx.x == 0) {
//         x0Idx = activeIdx_G[blockIdx.x];
//     }
//     x1Index = treeSize + blockIdx.x * blockDim.x;
//     __syncthreads();

//     if (!G[x0Idx]) {
//         return;
//     }

//     __shared__ float x0[SAMPLE_DIM];
//     __shared__ float x1s[BLOCK_SIZE * SAMPLE_DIM];
//     __shared__ float x1Conns[BLOCK_SIZE];
//     __shared__ int maxIndex[BLOCK_SIZE];
//     __shared__ float s_xGoal[SAMPLE_DIM];

//     if (threadIdx.x < sampleDim) {
//         x0[threadIdx.x] = samples[x0Idx * sampleDim + threadIdx.x];
//         s_xGoal[threadIdx.x] = xGoal[threadIdx.x];
//     }
//     __syncthreads();

//     float* x1 = &x1s[threadIdx.x * sampleDim];
//     curandState state = states[tid];
//     propagateState(x0, x1, numDisc, agentLength, &state);
//     states[tid] = state;
//     x1Conns[threadIdx.x] = calculateConnectivity(x1, s_xGoal);
//     maxIndex[threadIdx.x] = threadIdx.x;
//     __syncthreads();

//     for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
//         if (threadIdx.x < stride) {
//             float lhs = x1Conns[threadIdx.x];
//             float rhs = x1Conns[threadIdx.x + stride];
//             if (lhs < rhs) {
//                 x1Conns[threadIdx.x] = rhs;
//                 maxIndex[threadIdx.x] = maxIndex[threadIdx.x + stride];
//             }
//         }
//         __syncthreads();
//     }

//     if (threadIdx.x < sampleDim) {
//         samples[x1Index * sampleDim + threadIdx.x] = x1s[maxIndex[0] * sampleDim + threadIdx.x];
//     } else if (threadIdx.x == sampleDim) {
//         eParentIDx[x1Index] = x0Idx;
//     } else if (threadIdx.x == sampleDim + 1) {
//         eClosed[x0Idx] = true;  // TODO: Possibly remove eClosed and just use G.
//         G[x0Idx] = false;
//     } else if (threadIdx.x == sampleDim + 2) {
//         eUnexplored[x1Index] = true;
//     } else if (threadIdx.x == sampleDim + 3) {
//         eConn[x1Index] = x1Conns[maxIndex[0]];
//     }
// }



__global__
void expandEOpen(bool* eUnexplored, bool* eClosed, bool* eOpen, float* eConn, int* activeEUnexplored_Idx, int size_activeEUnexplored, float connThresh){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size_activeEUnexplored)
        return;
    int xIDx = activeEUnexplored_Idx[tid];
    if(eUnexplored[xIDx]){
        if(eConn[xIDx] > connThresh){
            eOpen[xIDx] = true;
            eUnexplored[xIDx] = false;
            return;
        }
        eUnexplored[xIDx] = false;
        eClosed[xIDx] = true;
    }
}

// TODO: Make this only add certain number of samples to G. S.T each block can handle prop of a single g in G.
__global__
void expandG(bool* eOpen, bool* G, float* eConn, int* activeEOpen_Idx, int size_activeEOpen, float connThresh){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size_activeEOpen)
        return;
    int xIDx = activeEOpen_Idx[tid];
    if(eOpen[xIDx]){
        if(eConn[xIDx] > connThresh){
            G[xIDx] = true;
            eOpen[xIDx] = false;
            return;
        }
    }
}

__global__ void initCurandStates(curandState* states, int numStates, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStates)
        return;
    curand_init(seed, tid, 0, &states[tid]);
}

__device__
void propagateState(float* x0, float* x1, int numDisc, float agentLength, curandState* state){
    // Generate random controls
    float a = curand_uniform(state) * 1.0f - .5f;
    float steering = curand_uniform(state) * M_PI - M_PI / 2; 
    float duration = curand_uniform(state) * 0.3f  + .05f; 

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
        theta += (v / agentLength) * tan_steering * dt;
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

// TODO: UPDATE THIS
// input is a sample x, output is the connectivity of the sample.
// determines a worth heuristic for the sample based on coverage.
__device__ float calculateConnectivity(float* x, float* xGoal){
    float dist = 0.0f;
    for (int i = 0; i < STATE_DIM - 2; ++i) { // TODO: currently only checking x and y
        dist += fabsf(x[i] - xGoal[i]);
    }
    return 100.0f - dist; // TODO: make this not 100.
}
