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

KGMT::KGMT(float width, float height, int N, int numIterations, int maxTreeSize, int maxSampleSize, int numDisc, int sampleDim, float agentLength):
    width_(width), height_(height), N_(N), numIterations_(numIterations), maxTreeSize_(maxTreeSize), maxSampleSize_(maxSampleSize), numDisc_(numDisc), sampleDim_(sampleDim), agentLength_(agentLength){

    d_eOpen_ = thrust::device_vector<bool>(maxTreeSize);
    d_eClosed_ = thrust::device_vector<bool>(maxTreeSize);
    d_G_ = thrust::device_vector<bool>(maxTreeSize);
    d_eUnexplored_ = thrust::device_vector<bool>(maxTreeSize);
    d_activeU_ = thrust::device_vector<bool>(maxTreeSize);
    d_edges_ = thrust::device_vector<int>(maxTreeSize);
    d_scanIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_activeGIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_activeIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_eParentIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_samples_ = thrust::device_vector<float>(maxTreeSize * sampleDim);
    d_eConnectivity_ = thrust::device_vector<float>(maxTreeSize);
    d_xGoal_ = thrust::device_vector<float>(sampleDim);
    d_uSamples_ = thrust::device_vector<float>(maxSampleSize * sampleDim);
    d_uParentIdx_ = thrust::device_vector<int>(maxSampleSize);
    d_uConn_ = thrust::device_vector<float>(maxSampleSize);

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
    d_uSamples_ptr_ = thrust::raw_pointer_cast(d_uSamples_.data());
    d_uParentIdx_ptr_ = thrust::raw_pointer_cast(d_uParentIdx_.data());
    d_uConn_ptr_ = thrust::raw_pointer_cast(d_uConn_.data());
    d_activeU_ptr_ = thrust::raw_pointer_cast(d_activeU_.data());
    

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
	const int gridSize = std::min((maxTreeSize_ + blockSize - 1) / blockSize, 2147483647);
    int gridSizeActive = 1;
    const int blockSizeActive = 32;

    // initialize random seed for curand
    curandState* d_states;
    cudaMalloc(&d_states, maxTreeSize_ * sizeof(curandState));
    initCurandStates<<<(maxTreeSize_ + blockSize - 1) / blockSize, blockSize>>>(d_states, maxTreeSize_, time(NULL));

    int itr = 0;
    treeSize_ = 1;
    int activeSize = 0;

    while(itr < numIterations_){
        itr++;
        // Propagate G:
        thrust::exclusive_scan(d_G_.begin(), d_G_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxTreeSize_-1];
        (d_G_[maxTreeSize_ - 1]) ? ++activeSize : 0;
        findInd<<<gridSize, blockSize>>>(maxTreeSize_, d_G_ptr_, d_scanIdx_ptr_, d_activeIdx_G_ptr_);
        gridSizeActive = std::min(activeSize, 32);
        propagateG<<<gridSizeActive, blockSizeActive>>>(d_xGoal_ptr_, d_uSamples_ptr_, d_samples_ptr_, d_activeU_ptr_, d_eClosed_ptr_, d_G_ptr_, d_uConn_ptr_, d_uParentIdx_ptr_, d_activeIdx_G_ptr_, activeSize, treeSize_, sampleDim_, agentLength_, numDisc_, d_states, connThresh_);

        // Find New G:
        thrust::exclusive_scan(d_activeU_.begin(), d_activeU_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxTreeSize_-1];
        (d_activeU_[maxTreeSize_ - 1]) ? ++activeSize : 0;
        findInd<<<gridSize, blockSize>>>(maxTreeSize_, d_activeU_ptr_, d_scanIdx_ptr_, d_activeIdx_ptr_);
        gridSizeActive = std::min((activeSize + blockSizeActive - 1) / blockSizeActive, 2147483647);
        expandG<<<gridSizeActive, blockSizeActive>>>(d_samples_ptr_, d_uSamples_ptr_, d_activeIdx_ptr_, d_uParentIdx_ptr_, d_eParentIdx_ptr_, d_G_ptr_, d_activeU_ptr_, activeSize, treeSize_);
        treeSize_ += gridSizeActive * blockSizeActive;

        // treeSize_ += 1;
        cudaMemcpy(&costGoal_, d_costGoal, sizeof(float), cudaMemcpyDeviceToHost);
    }

    double t_kgmt = (std::clock() - t_kgmtStart) / (double) CLOCKS_PER_SEC;
    std::cout << "time inside KGMT is " << t_kgmt << std::endl;

    // move vectors to csv to be plotted.
    copyAndWriteVectorToCSV(d_samples_, "samples.csv", maxTreeSize_, sampleDim_);
    copyAndWriteVectorToCSV(d_eParentIdx_, "parentRelations.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_eUnexplored_, "unexplored.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_eOpen_, "open.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_eClosed_, "closed.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_G_, "G.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_eConnectivity_, "connectivity.csv", maxTreeSize_, 1);

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

// Extends blockDim.x samples per sample in G.
__global__
void propagateG(float* xGoal, float* uSamples, float* samples, bool* activeU, bool* eClosed, bool* G, float* uConn, int* uParentIdx, int* activeIdx_G, int activeSize_G, int treeSize, int sampleDim, float agentLength, int numDisc, curandState* states, float connThresh) {
    
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
        float* x1 = &uSamples[tid * sampleDim];
        curandState state = states[tid];
        propagateState(x0, x1, numDisc, agentLength, &state);
        uConn[tid] = calculateConnectivity(x1, s_xGoal);
        states[tid] = state;
        if(uConn[tid] > connThresh){
            uParentIdx[tid] = x0Idx;
            activeU[tid] = true;
        }
        if (threadIdx.x == 0) {
            eClosed[x0Idx] = true;  // TODO: Possibly remove eClosed and just use G.
            G[x0Idx] = false;
        }
    }
}

__global__ 
void expandG(float* samples, float* uSamples, int* activeIdx, int* uParentIdx, int* tParentIdx, bool* G, bool* activeU, int activeSize, int treeSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= activeSize)
        return;
    int xIDx = activeIdx[tid];
    if(activeU[xIDx] ){
        float* x0 = &samples[uParentIdx[xIDx]*SAMPLE_DIM];
        float* x1 = &uSamples[xIDx*SAMPLE_DIM];
        if(!inCollision(x0, x1)){
            int x1Idx = treeSize + tid;
            for (int i = 0; i < SAMPLE_DIM; i++) {
                samples[x1Idx * SAMPLE_DIM + i] = x1[i];
            }
            G[x1Idx] = true;
            activeU[xIDx] = false;
            tParentIdx[x1Idx] = uParentIdx[xIDx];
        }
    }
}


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

// TODO: UPDATE THIS
__device__ bool inCollision(float* x0, float* x1){
    return false;
}
