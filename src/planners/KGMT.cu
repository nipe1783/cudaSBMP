#include "planners/KGMT.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "agent/Agent.h"
#include "state/State.h"
#include "helper/helper.cuh"
#include "helper/helper.cu"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <chrono>
#include <ctime>
#include <cub/cub.cuh>

#define SAMPLE_DIM 7
#define STATE_DIM 4
#define BLOCK_SIZE 32
#define NUM_R2 16

KGMT::KGMT(float width, float height, int N, int n, int numIterations, int maxTreeSize, int maxSampleSize, int numDisc, int sampleDim, float agentLength):
    width_(width), height_(height), N_(N), n_(n), numIterations_(numIterations), maxTreeSize_(maxTreeSize), maxSampleSize_(maxSampleSize), numDisc_(numDisc), sampleDim_(sampleDim), agentLength_(agentLength){

    R1Size_ = width / N;
    R2Size_ = width / (n*N);

    d_G_ = thrust::device_vector<bool>(maxTreeSize);
    d_U_ = thrust::device_vector<bool>(maxTreeSize);
    d_scanIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_UscanIdx_= thrust::device_vector<int>(maxSampleSize);
    d_R1scanIdx_ = thrust::device_vector<int>(N*N);
    d_activeIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_activeUIdx_ = thrust::device_vector<int>(maxSampleSize);
    d_activeR1Idx_ = thrust::device_vector<int>(N*N);
    d_treeParentIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_treeSamples_ = thrust::device_vector<float>(maxTreeSize * sampleDim);
    d_xGoal_ = thrust::device_vector<float>(sampleDim);
    d_unexploredSamples_ = thrust::device_vector<float>(maxSampleSize * sampleDim);
    d_uParentIdx_ = thrust::device_vector<int>(maxSampleSize);
    d_R1Avail_ = thrust::device_vector<int>(N*N);
    d_R2Avail_ = thrust::device_vector<int>(N*N*n*n);
    d_R1Score_ = thrust::device_vector<float>(N*N);
    d_R1Valid_ = thrust::device_vector<int>(N*N);
    d_R2Valid_ = thrust::device_vector<int>(N*N*n*n);
    d_R1Invalid_ = thrust::device_vector<int>(N*N);
    d_R2Invalid_ = thrust::device_vector<int>(N*N*n*n);
    d_R1_ = thrust::device_vector<int>(N*N);
    d_R2_ = thrust::device_vector<int>(N*N*n*n);
    d_R1Sel_ = thrust::device_vector<int>(N*N);
    d_uValid_ = thrust::device_vector<bool>(maxSampleSize);

    d_G_ptr_ = thrust::raw_pointer_cast(d_G_.data());
    d_treeSamples_ptr_ = thrust::raw_pointer_cast(d_treeSamples_.data());
    d_scanIdx_ptr_ = thrust::raw_pointer_cast(d_scanIdx_.data());
    d_UscanIdx_ptr_ = thrust::raw_pointer_cast(d_UscanIdx_.data());
    d_R1scanIdx_ptr_ = thrust::raw_pointer_cast(d_R1scanIdx_.data());
    d_activeIdx_ptr_ = thrust::raw_pointer_cast(d_activeIdx_.data());
    d_activeUIdx_ptr_ = thrust::raw_pointer_cast(d_activeUIdx_.data());
    d_activeR1Idx_ptr_ = thrust::raw_pointer_cast(d_activeR1Idx_.data());
    d_treeParentIdx_ptr_ = thrust::raw_pointer_cast(d_treeParentIdx_.data());
    d_xGoal_ptr_ = thrust::raw_pointer_cast(d_xGoal_.data());
    d_unexploredSamples_ptr_ = thrust::raw_pointer_cast(d_unexploredSamples_.data());
    d_uParentIdx_ptr_ = thrust::raw_pointer_cast(d_uParentIdx_.data());
    d_U_ptr_ = thrust::raw_pointer_cast(d_U_.data());
    d_R1Score_ptr_ = thrust::raw_pointer_cast(d_R1Score_.data());
    d_R1Avail_ptr_ = thrust::raw_pointer_cast(d_R1Avail_.data());
    d_R2Avail_ptr_ = thrust::raw_pointer_cast(d_R2Avail_.data());
    d_R1_ptr_ = thrust::raw_pointer_cast(d_R1_.data());
    d_R2_ptr_ = thrust::raw_pointer_cast(d_R2_.data());
    d_uValid_ptr_ = thrust::raw_pointer_cast(d_uValid_.data());
    d_R1Valid_ptr_ = thrust::raw_pointer_cast(d_R1Valid_.data());
    d_R2Valid_ptr_ = thrust::raw_pointer_cast(d_R2Valid_.data());
    d_R1Invalid_ptr_ = thrust::raw_pointer_cast(d_R1Invalid_.data());
    d_R2Invalid_ptr_ = thrust::raw_pointer_cast(d_R2Invalid_.data());
    d_R1Sel_ptr_ = thrust::raw_pointer_cast(d_R1Sel_.data());


    cudaMalloc(&d_costToGoal, sizeof(float));
    thrust::fill(d_treeParentIdx_.begin(), d_treeParentIdx_.end(), -1);
    thrust::fill(d_uParentIdx_.begin(), d_uParentIdx_.end(), -1);

}

void KGMT::plan(float* initial, float* goal) {
    
    double t_kgmtStart = std::clock();
    
    // initialize vectors with root of tree
    cudaMemcpy(d_treeSamples_ptr_, initial, sampleDim_ * sizeof(float), cudaMemcpyHostToDevice);
    bool value = true;
    cudaMemcpy(d_G_ptr_, &value, sizeof(bool), cudaMemcpyHostToDevice);

    // initialize xGoal
    cudaMemcpy(d_xGoal_ptr_, goal, sampleDim_ * sizeof(float), cudaMemcpyHostToDevice);
    
    const int blockSize = 128;
	const int gridSize = std::min((maxTreeSize_ + blockSize - 1) / blockSize, 2147483647);
    int gridSizeActive = 1;
    int blockSizeActive = 32;

    // initialize random seed for curand
    curandState* d_randomStates;
    cudaMalloc(&d_randomStates, maxTreeSize_ * sizeof(curandState));
    initCurandStates<<<(maxTreeSize_ + blockSize - 1) / blockSize, blockSize>>>(d_randomStates, maxTreeSize_, time(NULL));

    int itr = 0;
    treeSize_ = 1;
    int activeSize = 0;
    int maxIndex;
    float maxValue;
    while(itr < numIterations_){
        itr++;

        // Propagate G:
        thrust::exclusive_scan(d_G_.begin(), d_G_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxTreeSize_-1];
        (d_G_[maxTreeSize_ - 1]) ? ++activeSize : 0;
        findInd<<<gridSize, blockSize>>>(
            maxTreeSize_, 
            d_G_ptr_, 
            d_scanIdx_ptr_, 
            d_activeIdx_ptr_);
        gridSizeActive = std::min(activeSize, 32);
        propagateG<<<gridSizeActive, blockSizeActive>>>(
            activeSize, 
            d_activeIdx_ptr_, 
            d_G_ptr_, 
            d_U_ptr_,
            d_uValid_ptr_,
            d_treeSamples_ptr_, 
            d_unexploredSamples_ptr_,
            d_uParentIdx_ptr_,
            d_R1Valid_ptr_,
            d_R2Valid_ptr_,
            d_R1Invalid_ptr_,
            d_R2Invalid_ptr_,
            d_R1_ptr_,
            d_R2_ptr_,
            d_R1Avail_ptr_,
            d_R2Avail_ptr_,
            N_,
            n_,
            R1Size_,
            R2Size_,
            d_randomStates, 
            numDisc_, 
            agentLength_);

        // update sOpen:
        thrust::exclusive_scan(d_uValid_.begin(), d_uValid_.end(), d_UscanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_UscanIdx_[maxSampleSize_-1];
        (d_uValid_[maxSampleSize_ - 1]) ? ++activeSize : 0;
        findInd<<<gridSize, blockSize>>>(
            maxSampleSize_, 
            d_uValid_ptr_, 
            d_UscanIdx_ptr_, 
            d_activeIdx_ptr_);
        
        
        
        // Update Grid:
        // thrust::exclusive_scan(d_R1Avail_.begin(), d_R1Avail_.end(), d_R1scanIdx_.begin(), 0, thrust::plus<int>());
        // activeSize = d_R1scanIdx_[N_*N_-1];
        // (d_R1Avail_[N_*N_ - 1]) == 1 ? ++activeSize : 0;
        // findInd<<<gridSize, blockSize>>>(
        //     N_*N_, 
        //     d_R1Avail_ptr_, 
        //     d_R1scanIdx_ptr_, 
        //     d_activeR1Idx_ptr_);
        // gridSizeActive = activeSize;
        // blockSizeActive = n_*n_;
        // printDeviceVector(d_R2Avail_ptr_, N_*N_*n_*n_);
        gridSizeActive = N_*N_;
        blockSizeActive = n_*n_;
        updateR1<<<gridSizeActive, blockSizeActive>>>(
            d_R1Score_ptr_, 
            d_R1Avail_ptr_, 
            d_R2Avail_ptr_,
            d_R1Valid_ptr_,
            d_R1Invalid_ptr_,
            d_R1Sel_ptr_,
            n_, 
            0.1, 
            R2Size_*R2Size_);
        thrust::device_vector<float>::iterator iter = thrust::max_element(d_R1Score_.begin(), d_R1Score_.end());
        maxIndex = iter - d_R1Score_.begin();
        maxValue = *iter;

        // update G:

        
        
        cudaMemcpy(&costToGoal_, d_costToGoal, sizeof(float), cudaMemcpyDeviceToHost);

        // printf("R1 Avail: \n");
        // printDeviceVector(d_R1Avail_ptr_, 100);
        // printf("R2 Avail: \n");
        // printDeviceVector(d_R2Avail_ptr_, N_*N_*n_*n_);
        // printf("R1 Valid: \n");
        // printDeviceVector(d_R1Valid_ptr_, 100);
        // printf("R2 Valid: \n");
        // printDeviceVector(d_R2Valid_ptr_, 100);
        // printf("R1 Invalid: \n");
        // printDeviceVector(d_R1Invalid_ptr_, 100);
        // printf("R2 Invalid: \n");
        // printDeviceVector(d_R2Invalid_ptr_, 100);
        // printf("R1: \n");
        // printDeviceVector(d_R1_ptr_, 100);
        // printf("R2: \n");
        // printDeviceVector(d_R2_ptr_, 100);
        // printf("Valid U: \n");
        // printDeviceVector(d_uValid_ptr_, 100);

    }

    double t_kgmt = (std::clock() - t_kgmtStart) / (double) CLOCKS_PER_SEC;
    std::cout << "time inside KGMT is " << t_kgmt << std::endl;

    // move vectors to csv to be plotted.
    copyAndWriteVectorToCSV(d_treeSamples_, "samples.csv", maxTreeSize_, sampleDim_);
    copyAndWriteVectorToCSV(d_unexploredSamples_, "unexploredSamples.csv", maxSampleSize_, sampleDim_);
    copyAndWriteVectorToCSV(d_treeParentIdx_, "parentRelations.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_uParentIdx_, "uParentIdx.csv", maxSampleSize_, 1);
    copyAndWriteVectorToCSV(d_G_, "G.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_R2Avail_, "R2Avail.csv", N_*N_*n_*n_, 1);
    copyAndWriteVectorToCSV(d_R1Avail_, "R1Avail.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R1Valid_, "R1Valid.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R2Valid_, "R2Valid.csv", N_*N_*n_*n_, 1);
    copyAndWriteVectorToCSV(d_R1Invalid_, "R1Invalid.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R2Invalid_, "R2Invalid.csv", N_*N_*n_*n_, 1);
    copyAndWriteVectorToCSV(d_R1Score_, "R1Score.csv", N_*N_, 1);

    // Free the allocated memory for curand states
    cudaFree(d_randomStates);
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

__global__
void findInd(int numSamples, int* S, int* scanIdx, int* activeS){
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numSamples)
        return;
    if (!S[node]) {
        return;
    }
    activeS[scanIdx[node]] = node;
}


__global__ void propagateG(
    int sizeG, 
    int* activeGIdx, 
    bool* G,
    bool* U,
    bool* uValid,
    float* treeSamples,
    float* unexploredSamples,
    int* uParentIdx,
    int* R1Valid,
    int* R2Valid,
    int* R1Invalid,
    int* R2Invalid,
    int* R1,
    int* R2,
    int* R1Avail,
    int* R2Avail,
    int N,
    int n,
    float R1Size,
    float R2Size,
    curandState* randomStates,
    int numDisc,
    float agentLength) {

    // block expands x0 BLOCK_SIZE times.
    if (blockIdx.x >= sizeG)
        return;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int x0Idx;
    if(threadIdx.x == 0){
        x0Idx = activeGIdx[blockIdx.x];
    }
    __syncthreads();

    __shared__ float x0[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM){
        x0[threadIdx.x] = treeSamples[x0Idx * SAMPLE_DIM + threadIdx.x];
    }
    __syncthreads();
    
    curandState randState = randomStates[tid];
    float* x1 = &unexploredSamples[tid * SAMPLE_DIM];
    uParentIdx[tid] = x0Idx;
    bool valid = propagateAndCheck(x0, x1, numDisc, agentLength, &randState);
    int r1 = getR1(x1[0], x1[1], R1Size, N);
    int r2 = getR2(x1[0], x1[1], r1, R1Size, N , R2Size, n);
    U[tid] = true;
    atomicAdd(&R1[r1], 1);
    atomicAdd(&R2[r2], 1);
    if(valid){
        if(R1Avail[r1] == 0){
            atomicExch(&R1Avail[r1], 1);
        }
        if(R2Avail[r2] == 0){
            atomicExch(&R2Avail[r2], 1);
        }
        atomicAdd(&R2Valid[r2], 1);
        atomicAdd(&R1Valid[r1], 1);
        uValid[tid] = true;
    } else {
        atomicAdd(&R1Invalid[r1], 1);
        atomicAdd(&R2Invalid[r2], 1);
    }
    randomStates[tid] = randState;
}

__global__ void updateR1(
    float* R1Score, 
    int* R1Avail, 
    int* R2Avail, 
    int* R1Valid, 
    int* R1Invalid,
    int* R1Sel,
    int n, 
    float epsilon, 
    float R1Vol) {
    
    if(R1Avail[blockIdx.x] == 0)
        return;

    __shared__ int nValid;
    __shared__ int nInvalid;
    __shared__ int nR1;
    if(threadIdx.x == 0){
        nValid = R1Valid[blockIdx.x];
    }
    if(threadIdx.x == 1){
        nInvalid = R1Invalid[blockIdx.x];
    }
    if(threadIdx.x == 2){
        nR1 = R1Sel[blockIdx.x];
    }
    __syncthreads();
    typedef cub::BlockReduce<int, NUM_R2*NUM_R2> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage tempStorage;
    int threadData = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < n*n){
        threadData = R2Avail[tid];
    }
    int availR2 = BlockReduceT(tempStorage).Sum(threadData);
    if (threadIdx.x == 0) {
        float covR = float(availR2) / float(n*n);
        float freeVol = ((epsilon + nValid) / (epsilon + nValid + nInvalid));
        float score = pow(freeVol, 2) / ((1 + covR) * (1 + pow(nR1, 2)));
        R1Score[blockIdx.x] = score;
    }
}

__global__ void updateG(float* samples){

}

__global__ void updateOpen(float* unexploredSamples, float* treeSamples, int* uParentIDx, int* treeParentIdx, bool* O, bool* U, bool* activeUIdx, int treeSize, int activeSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= activeSize)
        return;
    int treeSamplesIdx = treeSize + tid * SAMPLE_DIM;
    int unexploredSamplesIdx = activeUIdx[tid] * SAMPLE_DIM;
    if (activeUIdx[tid]) {
        treeSamples[treeSamplesIdx] = unexploredSamples[unexploredSamplesIdx * SAMPLE_DIM];
        treeSamples[treeSamplesIdx + 1] = unexploredSamples[unexploredSamplesIdx * SAMPLE_DIM + 1];
        treeSamples[treeSamplesIdx + 2] = unexploredSamples[unexploredSamplesIdx * SAMPLE_DIM + 2];
        treeSamples[treeSamplesIdx + 3] = unexploredSamples[unexploredSamplesIdx * SAMPLE_DIM + 3];
        treeSamples[treeSamplesIdx + 4] = unexploredSamples[unexploredSamplesIdx * SAMPLE_DIM + 4];
        treeSamples[treeSamplesIdx + 5] = unexploredSamples[unexploredSamplesIdx * SAMPLE_DIM + 5];
        treeSamples[treeSamplesIdx + 6] = unexploredSamples[unexploredSamplesIdx * SAMPLE_DIM + 6];
        treeParentIdx[treeSize + tid] = uParentIDx[activeUIdx[tid]];
        O[treeSize + tid] = true;
    }
}

__global__ void initCurandStates(curandState* states, int numStates, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStates)
        return;
    curand_init(seed, tid, 0, &states[tid]);
}

__device__
bool propagateAndCheck(float* x0, float* x1, int numDisc, float agentLength, curandState* state){
    // Generate random controls
    float a = curand_uniform(state) * 1.5f - .1f;
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
    //TODO: Update this.
    return true;
}

__device__ int getR1(float x, float y, float R1Size, int N) {
    int cellX = static_cast<int>(x / R1Size);
    int cellY = static_cast<int>(y / R1Size);
    if (cellX >= 0 && cellX < N && cellY >= 0 && cellY < N) {
        return cellY * N + cellX;
    }
    return -1; // Invalid cell
}
__device__ int getR2(float x, float y, int r1, float R1Size, int N, float R2Size, int n) {
    if (r1 == -1) {
        return -1; // Invalid R1 cell, so R2 is also invalid
    }

    int cellY_R1 = r1 / N;
    int cellX_R1 = r1 % N;

    // Calculate the local coordinates within the R1 cell
    float localX = x - cellX_R1 * R1Size;
    float localY = y - cellY_R1 * R1Size;

    int cellX_R2 = static_cast<int>(localX / R2Size);
    int cellY_R2 = static_cast<int>(localY / R2Size);
    if (cellX_R2 >= 0 && cellX_R2 < n && cellY_R2 >= 0 && cellY_R2 < n) {
        int localR2 = cellY_R2 * n + cellX_R2;
        return r1 * (n * n) + localR2; // Flattened index
    }
    return -1; // Invalid subcell
}
