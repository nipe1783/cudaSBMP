#pragma once

#include "planners/Planner.cuh"
#include "occupancyMaps/OccupancyGrid.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <curand_kernel.h>

class KGMT
{
    public:
        // constructor
        KGMT() = default;
        KGMT(float width, float height, int N, int n, int numIterations, int maxTreeSize, int numDisc, int sampleDim, float agentLength);

        // methods
        void plan(float* initial, float* goal);

        // fields
        int numIterations_; // Number of iterations to run KGMT
        int maxTreeSize_; // Maximum number of samples to store. Similar to number of samples in PRM exceept it is initialized to 0s.
        int numDisc_; // Number of iterations when doing state propagation
        int sampleDim_; // Dimension of each sample
        int treeSize_; // Current size of the tree. Informs where in the d_samples_ vector to add new samples.
        float width_; // Width of the workspace
        float height_; // Height of the workspace
        float costToGoal_; // Cost of the goal state
        float agentLength_; // Length of the agent. Used in state propagation.
        float R1Threshold_;
        thrust::device_vector<bool> d_G_; // Set of samples to be expanded in current iteration.
        thrust::device_vector<bool> d_GNew_;
        thrust::device_vector<bool> d_U_;
        thrust::device_vector<bool> d_uValid_;
        thrust::device_vector<int> d_scanIdx_; // stores scan of G. ex: G = [0, 1, 0, 1, 1, 0, 1] -> scanIdx = [0,0,1,1,2,3,3]. Used to find active samples in G.
        thrust::device_vector<int> d_scanIdxGnew_;
        thrust::device_vector<int> d_R1scanIdx_;
        thrust::device_vector<int> d_activeIdx_;
        thrust::device_vector<int> d_activeR1Idx_;
        thrust::device_vector<int> d_activeUIdx_;
        thrust::device_vector<int> d_treeParentIdx_; // Stores parent sample idx. Ex: d_parentIdx[10] = 3. Parent of sample 3 is 10.
        thrust::device_vector<int> d_uParentIdx_; // stores parent indeces for current unexplored iteration.
        thrust::device_vector<float> d_treeSamples_; // Stores all samples. Size is maxSamples_ * sampleDim_.
        thrust::device_vector<float> d_unexploredSamples_; // all unexplored samples of current iteration.
        thrust::device_vector<float> d_xGoal_;
        thrust::device_vector<int> d_R1Valid_;
        thrust::device_vector<int> d_R2Valid_;
        thrust::device_vector<int> d_R1Invalid_;
        thrust::device_vector<int> d_R2Invalid_;
        thrust::device_vector<int> d_R2_;
        thrust::device_vector<int> d_R1_; // number of times a sample has been located in region R1i.
        thrust::device_vector<int> d_R1Avail_;
        thrust::device_vector<int> d_R2Avail_;
        thrust::device_vector<float> d_R1Score_; // expansion score of region R.

        bool *d_G_ptr_;
        bool *d_GNew_ptr_;
        bool *d_U_ptr_;
        bool *d_uValid_ptr_;
        int *d_scanIdx_ptr_;
        int *d_R1scanIdx_ptr_;
        int *d_scanIdxGnew_ptr_;
        int *d_activeIdx_ptr_;
        int *d_activeR1Idx_ptr_;
        int *d_activeUIdx_ptr_;
        int *d_treeParentIdx_ptr_;
        int *d_uParentIdx_ptr_;
        float *d_treeSamples_ptr_;
        float *d_xGoal_ptr_;
        float *d_unexploredSamples_ptr_;
        int* d_R1Avail_ptr_;
        int* d_R2Avail_ptr_;
        int* d_R1Valid_ptr_;
        int* d_R2Valid_ptr_;
        int* d_R1Invalid_ptr_;
        int* d_R2Invalid_ptr_;
        int* d_R1_ptr_;
        int* d_R2_ptr_;
        int* d_uR1_ptr_;
        int* d_uR2_ptr_;
        int* d_uR1Count_ptr_;
        int* d_uR1Idx_ptr_;
        float* d_R1Score_ptr_;
        float *d_costToGoal;
        float *d_R1Threshold_ptr_;

        // occupancy grid:
        int N_; // Number of cols/rows in the workspace grid
        int n_; // Number of cols/rows per sub region in grid.
        float R1Size_;
        float R2Size_;
        
        
};

// GPU kernels:

__global__ void findInd(int numSamples, bool* S, int* scanIdx, int* activeGIdx);
__global__ void findInd(int numSamples, int* S, int* scanIdx, int* activeGIdx);
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
    float* R1Threshold,
    float* R1Scores,
    int itr);

__global__ void updateR1(
    float* R1Score, 
    int* R1Avail, 
    int* R2Avail, 
    int* R1Valid, 
    int* R1Invalid,
    int* R1Sel,
    int n, 
    float epsilon, 
    float R1Vol,
    float* R1Threshold,
    int activeSize);

__global__ void updateG(
    float* treeSamples, 
    float* unexploredSamples, 
    int* unexploredParentIdx,
    int* treeParentIdx,
    bool* G,
    bool* GNew,
    int* GNewIdx, 
    int GNewSize, 
    int treeSize);

__global__ void initCurandStates(curandState* states, int numStates, int seed);


__host__ __device__ int getR1(float x, float y, float R1Size, int N);
__host__ __device__ int getR2(float x, float y, int r1, float R1Size, int N, float R2Size, int n);
__device__ bool propagateAndCheck(float* x0, float* x1, int numDisc, float agentLength, curandState* state);

