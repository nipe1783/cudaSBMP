#pragma once
#include "helper/helper.cuh"
#include "collisionCheck/collisionCheck.cuh"
#include "statePropagator/statePropagator.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include "occupancyMaps/OccupancyGrid.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdio>
#include <Eigen/Core>
#include "agent/Agent.h"
#include "state/State.h"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include "sssp/sssp.cuh"

class GoalBiasedKGMT
{
    public:
        // constructor
        GoalBiasedKGMT() = default;
        GoalBiasedKGMT(float width, float height, int N, int n, int numIterations, int maxTreeSize, int numDisc, float agentLength, float goalThreshold);

        // methods
        void plan(float* initial, float* goal, float *d_obstacles, int obstaclesCount);

        // fields
        int numIterations_; // Number of iterations to run GoalBiasedKGMT
        int maxTreeSize_; // Maximum number of samples to store. Similar to number of samples in PRM exceept it is initialized to 0s.
        int numDisc_; // Number of iterations when doing state propagation
        int treeSize_; // Current size of the tree. Informs where in the d_samples_ vector to add new samples.
        float width_; // Width of the workspace
        float height_; // Height of the workspace
        float costToGoal_; // Cost of the goal state
        float agentLength_; // Length of the agent. Used in state propagation.
        float R1Threshold_;
        float goalThreshold_;
        int nR1Edges_;
        int tableSize_;
        bool* d_finished_ptr_;
        thrust::device_vector<bool> d_G_; // Set of samples to be expanded in current iteration.
        thrust::device_vector<bool> d_GNew_;
        thrust::device_vector<int> d_scanIdx_; // stores scan of G. ex: G = [0, 1, 0, 1, 1, 0, 1] -> scanIdx = [0,0,1,1,2,3,3]. Used to find active samples in G.
        thrust::device_vector<int> d_R1scanIdx_;
        thrust::device_vector<int> d_activeIdx_;
        thrust::device_vector<int> d_activeR1Idx_;
        thrust::device_vector<int> d_treeParentIdx_; // Stores parent sample idx. Ex: d_parentIdx[10] = 3. Parent of sample 3 is 10.
        thrust::device_vector<int> d_uParentIdx_; // stores parent indices for current unexplored iteration.
        thrust::device_vector<float> d_treeSamples_; // Stores all samples. Size is maxSamples_ * sampleDim_.
        thrust::device_vector<float> d_unexploredSamples_; // all unexplored samples of current iteration.
        thrust::device_vector<float> d_xGoal_;
        thrust::device_vector<int> d_R1Valid_;
        thrust::device_vector<int> d_R2Valid_;
        thrust::device_vector<int> d_R2_;
        thrust::device_vector<int> d_R1_; // number of times a sample has been located in region R1i.
        thrust::device_vector<int> d_R1Avail_;
        thrust::device_vector<int> d_R2Avail_;
        thrust::device_vector<float> d_R1Score_; // expansion score of region R.
        thrust::device_vector<float> d_costs_;
        thrust::device_vector<float> d_R1EdgeCosts_;
        thrust::device_vector<int> d_selR1Edge_;
        thrust::device_vector<float> d_connR1Edge_;
        thrust::device_vector<int> d_fromR1_;
        thrust::device_vector<int> d_toR1_;
        thrust::device_vector<int> d_valR1Edge_;
        thrust::device_vector<int> d_fromNodes_; // fromNodes vector for edges
        thrust::device_vector<int> d_toNodes_; // toNodes vector for edges
        thrust::device_vector<int> d_edgeIndices_; // edge indices vector
        thrust::device_vector<int> d_hashTable_; // hash table for fast edge look-up
        thrust::device_vector<float> d_R1Dists_;

        bool *d_G_ptr_;
        bool *d_GNew_ptr_;
        bool *d_U_ptr_;
        bool *d_uValid_ptr_;
        int *d_scanIdx_ptr_;
        int *d_R1scanIdx_ptr_;
        int *d_scanIdxGnew_ptr_;
        int *d_activeIdx_ptr_;
        int *d_activeR1Idx_ptr_;
        int *d_treeParentIdx_ptr_;
        int *d_uParentIdx_ptr_;
        float *d_treeSamples_ptr_;
        float *d_xGoal_ptr_;
        float *d_unexploredSamples_ptr_;
        int* d_R1Avail_ptr_;
        int* d_R2Avail_ptr_;
        int* d_R1Valid_ptr_;
        int* d_R2Valid_ptr_;
        int* d_R1_ptr_;
        int* d_R2_ptr_;
        int* d_uR1_ptr_;
        int* d_uR2_ptr_;
        int* d_uR1Count_ptr_;
        int* d_uR1Idx_ptr_;
        float* d_R1Score_ptr_;
        float *d_costToGoal;
        float *d_R1Threshold_ptr_;
        float* d_costs_ptr_;
        float* d_R1EdgeCosts_ptr_;
        int* d_selR1Edge_ptr_;
        float* d_connR1Edge_ptr_;
        int* d_fromR1_ptr_;
        int* d_toR1_ptr_;
        int* d_valR1Edge_ptr_;
        int* d_fromNodes_ptr_;
        int* d_toNodes_ptr_;
        int* d_edgeIndices_ptr_;
        int* d_hashTable_ptr_;
        float* d_R1Dists_ptr_;


        // occupancy grid:
        int N_; // Number of cols/rows in the workspace grid
        int n_; // Number of cols/rows per sub region in grid.
        float R1Size_;
        float R2Size_;
        
};


// GoalBiasedKGMT kernels:
__global__ void findInd_gb(int numSamples, bool* S, int* scanIdx, int* activeGIdx);
__global__ void findInd_gb(int numSamples, int* S, int* scanIdx, int* activeGIdx);
__global__ void propagateG_gb(
    int sizeG, 
    int* activeGIdx, 
    bool* G,
    bool* GNew,
    float* treeSamples,
    float* unexploredSamples,
    int* uParentIdx,
    int* R1Valid,
    int* R2Valid,
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
    float* obstacles,
    int obstaclesCount,
    float width,
    float height,
    int* selR1Edge,
    int* valR1Edge,
    int* hashTable,
    int tableSize);

__global__ void propagateGV2_gb(
    int sizeG, 
    int* activeGIdx, 
    bool* G,
    bool* GNew,
    float* treeSamples,
    float* unexploredSamples,
    int* uParentIdx,
    int* R1Valid,
    int* R2Valid,
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
    float* obstacles,
    int obstacleCount,
    int iterations,
    float width,
    float height,
    int* selR1Edge,
    int* valR1Edge);

__global__ void updateR1_gb(
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

__global__ void updateR_gb(
    float* R1Score,
    int* R2Avail,
    int* R1Avail,  
    int* R1Valid,
    int* R1,
    int n, 
    float epsilon, 
    float* R1EdgeCosts,
    int activeSize,
    int* fromR1,
    int* toR1,
    int nR1Edges,
    int* selR1Edge,
    int* valR1Edge,
    float* R1Threshold);

__global__ void updateG_gb(
        float* treeSamples, 
    float* unexploredSamples, 
    int* uParentIdx,
    int* treeParentIdx,
    bool* G,
    bool* GNew,
    int* GNewIdx, 
    int GNewSize, 
    int treeSize,
    float* costs,
    float* xGoal,
    float r,
    float* costToGoal);

__global__ void initCurandStates_gb(curandState* states, int numStates, int seed);
__device__ float getCost_gb(float* x0, float* x1);
__device__ bool inGoalRegion_gb(float* x, float* goal, float r);


