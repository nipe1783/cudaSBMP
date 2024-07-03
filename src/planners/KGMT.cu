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
#include <filesystem>


#define SAMPLE_DIM 7
#define STATE_DIM 4
#define BLOCK_SIZE 32
#define NUM_R2 16
#define NUM_R1 16

KGMT::KGMT(float width, float height, int N, int n, int numIterations, int maxTreeSize, int numDisc, int sampleDim, float agentLength):
    width_(width), height_(height), N_(N), n_(n), numIterations_(numIterations), maxTreeSize_(maxTreeSize), numDisc_(numDisc), sampleDim_(sampleDim), agentLength_(agentLength){

    R1Size_ = width / N;
    R2Size_ = width / (n*N);

    d_G_ = thrust::device_vector<bool>(maxTreeSize);
    d_GNew_ = thrust::device_vector<bool>(maxTreeSize);
    d_U_ = thrust::device_vector<bool>(maxTreeSize);
    d_scanIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_scanIdxGnew_= thrust::device_vector<int>(maxTreeSize);
    d_R1scanIdx_ = thrust::device_vector<int>(N*N);
    d_activeIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_activeUIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_activeR1Idx_ = thrust::device_vector<int>(N*N);
    d_treeParentIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_treeSamples_ = thrust::device_vector<float>(maxTreeSize * sampleDim);
    d_xGoal_ = thrust::device_vector<float>(sampleDim);
    d_unexploredSamples_ = thrust::device_vector<float>(maxTreeSize * sampleDim);
    d_uParentIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_R1Avail_ = thrust::device_vector<int>(N*N);
    d_R2Avail_ = thrust::device_vector<int>(N*N*n*n);
    d_R1Score_ = thrust::device_vector<float>(N*N);
    d_R1Valid_ = thrust::device_vector<int>(N*N);
    d_R2Valid_ = thrust::device_vector<int>(N*N*n*n);
    d_R1Invalid_ = thrust::device_vector<int>(N*N);
    d_R2Invalid_ = thrust::device_vector<int>(N*N*n*n);
    d_R1_ = thrust::device_vector<int>(N*N);
    d_R2_ = thrust::device_vector<int>(N*N*n*n);
    d_uValid_ = thrust::device_vector<bool>(maxTreeSize);

    d_G_ptr_ = thrust::raw_pointer_cast(d_G_.data());
    d_GNew_ptr_ = thrust::raw_pointer_cast(d_GNew_.data());
    d_treeSamples_ptr_ = thrust::raw_pointer_cast(d_treeSamples_.data());
    d_scanIdx_ptr_ = thrust::raw_pointer_cast(d_scanIdx_.data());
    d_scanIdxGnew_ptr_ = thrust::raw_pointer_cast(d_scanIdxGnew_.data());
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


    cudaMalloc(&d_costToGoal, sizeof(float));
    thrust::fill(d_treeParentIdx_.begin(), d_treeParentIdx_.end(), -1);
    thrust::fill(d_uParentIdx_.begin(), d_uParentIdx_.end(), -1);
    thrust::fill(d_R1Score_.begin(), d_R1Score_.end(), 1.0);

}

void KGMT::plan(float* initial, float* goal) {
    
    double t_kgmtStart = std::clock();
    
    // initialize vectors with root of tree
    cudaMemcpy(d_treeSamples_ptr_, initial, sampleDim_ * sizeof(float), cudaMemcpyHostToDevice);
    bool value = true;
    cudaMemcpy(d_G_ptr_, &value, sizeof(bool), cudaMemcpyHostToDevice);
    int r1_0 = getR1(initial[0], initial[1], R1Size_, N_);
    int r2_0 = getR2(initial[0], initial[1], r1_0, R1Size_, N_, R2Size_, n_);
    thrust::device_ptr<int> d_R1_ptr = d_R1_.data();
    thrust::device_ptr<int> d_R1Avail_ptr = d_R1Avail_.data();
    thrust::device_ptr<int> d_R2Avail_ptr = d_R2Avail_.data();
    thrust::device_ptr<int> d_R1Valid_ptr = d_R1Valid_.data();
    thrust::fill(d_R1_ptr + r1_0, d_R1_ptr + r1_0 + 1, 1);
    thrust::fill(d_R1Avail_ptr + r1_0, d_R1Avail_ptr + r1_0 + 1, 1);
    thrust::fill(d_R2Avail_ptr + r2_0, d_R2Avail_ptr + r2_0 + 1, 1);
    thrust::fill(d_R1Valid_ptr + r1_0, d_R1Valid_ptr + r1_0 + 1, 1);
    R1Threshold_ = 0.01;

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

        // UPDATE GRID SCORES:
        thrust::exclusive_scan(d_R1Avail_.begin(), d_R1Avail_.end(), d_R1scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_R1scanIdx_[N_*N_-1];
        (d_R1Avail_[N_*N_ - 1]) == 1 ? ++activeSize : 0;
        findInd<<<gridSize, blockSize>>>(
            N_*N_, 
            d_R1Avail_ptr_, 
            d_R1scanIdx_ptr_, 
            d_activeR1Idx_ptr_);
        updateR1<<<1, N_*N_>>>(
            d_R1Score_ptr_, 
            d_R1Avail_ptr_, 
            d_R2Avail_ptr_,
            d_R1Valid_ptr_,
            d_R1Invalid_ptr_,
            d_R1_ptr_,
            n_, 
            0.01, 
            R2Size_*R2Size_);

        // PROPAGATE G:
        thrust::exclusive_scan(d_G_.begin(), d_G_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxTreeSize_-1];
        (d_G_[maxTreeSize_ - 1]) ? ++activeSize : 0;
        
        findInd<<<gridSize, blockSize>>>(
            maxTreeSize_, 
            d_G_ptr_, 
            d_scanIdx_ptr_, 
            d_activeIdx_ptr_);

        
        blockSizeActive = 32;
        gridSizeActive = std::min(activeSize, int(floor(maxTreeSize_ / blockSizeActive)));
        propagateG<<<gridSizeActive, blockSizeActive>>>(
            activeSize, 
            d_activeIdx_ptr_, 
            d_G_ptr_,
            d_GNew_ptr_,
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
            agentLength_,
            R1Threshold_,
            d_R1Score_ptr_,
            itr);
        
        // UPDATE G:
        thrust::exclusive_scan(d_GNew_.begin(), d_GNew_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxTreeSize_-1];
        (d_GNew_[maxTreeSize_ - 1]) ? ++activeSize : 0;
        findInd<<<gridSize, blockSize>>>(
            maxTreeSize_, 
            d_GNew_ptr_, 
            d_scanIdx_ptr_, 
            d_activeIdx_ptr_);
        gridSizeActive = std::min(activeSize, int(floor(maxTreeSize_ / blockSizeActive)));
        blockSizeActive = 128;
        updateG<<<gridSizeActive, blockSizeActive>>>(
            d_treeSamples_ptr_, 
            d_unexploredSamples_ptr_, 
            d_uParentIdx_ptr_,
            d_treeParentIdx_ptr_,
            d_G_ptr_,
            d_GNew_ptr_,
            d_activeIdx_ptr_, 
            activeSize, 
            treeSize_);
        
        treeSize_ += activeSize;

        // printf("treeSize is %d\n", treeSize_);
        cudaMemcpy(&costToGoal_, d_costToGoal, sizeof(float), cudaMemcpyDeviceToHost);
        if(treeSize_ >= maxTreeSize_){
            // printf("Tree size exceeded maxTreeSize\n");
            break;
        }

        
        std::ostringstream filename;
        std::filesystem::create_directories("Data");
        std::filesystem::create_directories("Data/Samples");
        std::filesystem::create_directories("Data/Parents");
        std::filesystem::create_directories("Data/R1Scores");
        std::filesystem::create_directories("Data/R1Avail");
        std::filesystem::create_directories("Data/R1");
        filename.str("");
        filename << "Data/Samples/samples" << itr << ".csv";
        copyAndWriteVectorToCSV(d_treeSamples_, filename.str(), maxTreeSize_, sampleDim_);
        filename.str("");
        filename << "Data/Parents/parents" << itr << ".csv";
        copyAndWriteVectorToCSV(d_treeParentIdx_, filename.str(), maxTreeSize_, 1);
        filename.str("");
        filename << "Data/R1Scores/R1Scores" << itr << ".csv";
        copyAndWriteVectorToCSV(d_R1Score_, filename.str(), N_*N_, 1);
        filename.str("");
        filename << "Data/R1Avail/R1Avail" << itr << ".csv";
        copyAndWriteVectorToCSV(d_R1Avail_, filename.str(), N_*N_, 1);
        filename.str("");
        filename << "Data/R1/R1" << itr << ".csv";
        copyAndWriteVectorToCSV(d_R1_, filename.str(), N_*N_, 1);
    }

    double t_kgmt = (std::clock() - t_kgmtStart) / (double) CLOCKS_PER_SEC;
    std::cout << "time inside KGMT is " << t_kgmt << std::endl;

    // move vectors to csv to be plotted.
    copyAndWriteVectorToCSV(d_treeSamples_, "samples.csv", maxTreeSize_, sampleDim_);
    copyAndWriteVectorToCSV(d_unexploredSamples_, "unexploredSamples.csv", maxTreeSize_, sampleDim_);
    copyAndWriteVectorToCSV(d_treeParentIdx_, "parentRelations.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_uParentIdx_, "uParentIdx.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_G_, "G.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_R2Avail_, "R2Avail.csv", N_*N_*n_*n_, 1);
    copyAndWriteVectorToCSV(d_R1Avail_, "R1Avail.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R1Valid_, "R1Valid.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R2Valid_, "R2Valid.csv", N_*N_*n_*n_, 1);
    copyAndWriteVectorToCSV(d_R1Invalid_, "R1Invalid.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R2Invalid_, "R2Invalid.csv", N_*N_*n_*n_, 1);
    copyAndWriteVectorToCSV(d_R1Score_, "R1Score.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R1_, "R1.csv", N_*N_, 1);

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
    bool* GNew,
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
    float agentLength,
    float R1Threshold,
    float* R1Scores,
    int itr) {

    // block expands x0 BLOCK_SIZE times.
    if (blockIdx.x >= sizeG)
        return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int x0Idx;
    if(threadIdx.x == 0){
        x0Idx = activeGIdx[blockIdx.x];
        G[x0Idx] = false;
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
    atomicAdd(&R1[r1], 1);
    atomicAdd(&R2[r2], 1);
    if(valid){
        if(R1Scores[r1] > R1Threshold){
            GNew[tid] = true;
        }
        if(R1Avail[r1] == 0){
            atomicExch(&R1Avail[r1], 1);
        }
        if(R2Avail[r2] == 0){
            atomicExch(&R2Avail[r2], 1);
        }
        atomicAdd(&R2Valid[r2], 1);
        atomicAdd(&R1Valid[r1], 1);
    }
    else {
        atomicAdd(&R1Invalid[r1], 1);
        atomicAdd(&R2Invalid[r2], 1);
    }
    randomStates[tid] = randState;

}

// 1 Block Version. Each thread calculates 1 R1 cell.
// TODO: Change it to a 2D block. each thread square calculates 1 R1 cell. Should help with fetching R2Avail.
__global__ void updateR1(
    float* R1Score, 
    int* R1Avail, 
    int* R2Avail, 
    int* R1Valid, 
    int* R1Invalid,
    int* R1,
    int n, 
    float epsilon, 
    float R1Vol) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_R1 * NUM_R1)
        return;

    // Use shared memory for reduction
    __shared__ float s_totalSum;

    float score = 0.0f;
    if (R1Avail[tid] != 0) {
        int nValid = R1Valid[tid];
        float covR = 0;
        for (int i = tid * n * n; i < (tid + 1) * n * n; i++) {
            covR += R2Avail[i];
        }
        covR /= n * n;

        float freeVol = ((epsilon + nValid) / (epsilon + nValid + R1Invalid[tid]));
        score = pow(freeVol, 4) / ((1 + covR) * (1 + pow(R1[tid], 2)));
    }

    typedef cub::BlockReduce<float, NUM_R1*NUM_R1> BlockReduceFloatT;
    __shared__ typename BlockReduceFloatT::TempStorage tempStorageFloat;
    float blockSum = BlockReduceFloatT(tempStorageFloat).Sum(score);

    if (threadIdx.x == 0) {
        s_totalSum = blockSum;
    }
    __syncthreads();

    // Normalize the score
    if(R1Avail[tid] == 0){
        R1Score[tid] = 1.0f;
    }
    else {
        R1Score[tid] = score / s_totalSum;
    }
}

__global__ void updateG(
    float* treeSamples, 
    float* unexploredSamples, 
    int* unexploredParentIdx,
    int* treeParentIdx,
    bool* G,
    bool* GNew,
    int* GNewIdx, 
    int GNewSize, 
    int treeSize){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= GNewSize)
        return;

    // move valid unexplored sample to tree:
    int x1TreeIdx = treeSize + tid;
    int x1UnexploredIdx = GNewIdx[tid];
    treeParentIdx[x1TreeIdx] = unexploredParentIdx[x1UnexploredIdx];
    treeSamples[x1TreeIdx * SAMPLE_DIM] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 1] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 1];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 2] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 2];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 3] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 3];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 4] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 4];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 5] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 5];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 6] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 6];

    // update G:
    G[x1TreeIdx] = true;
    GNew[tid] = false;
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
    float a = curand_uniform(state) * 20.0f - 10.0f;  // a between -5 and 5
    float steering = curand_uniform(state) * 2.0f * M_PI - M_PI;  // steering between -π and π
    float duration = curand_uniform(state) * .4f + 0.1f;  // duration between 0.1 and 0.5

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

__host__ __device__ int getR1(float x, float y, float R1Size, int N) {
    int cellX = static_cast<int>(x / R1Size);
    int cellY = static_cast<int>(y / R1Size);
    if (cellX >= 0 && cellX < N && cellY >= 0 && cellY < N) {
        return cellY * N + cellX;
    }
    return -1; // Invalid cell
}
__host__ __device__ int getR2(float x, float y, int r1, float R1Size, int N, float R2Size, int n) {
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
