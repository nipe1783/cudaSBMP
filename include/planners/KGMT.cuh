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
        KGMT(float width, float height, int N, int numIterations, int maxTreeSize, int maxSampleSize, int numDisc, int sampleDim, float agentLength);

        // methods
        void plan(float* initial, float* goal);

        // fields
        int numIterations_; // Number of iterations to run KGMT
        int maxTreeSize_; // Maximum number of samples to store. Similar to number of samples in PRM exceept it is initialized to 0s.
        int maxSampleSize_; // Maximum number of samples to store. Similar to number of samples in PRM exceept it is initialized to 0s.
        int numDisc_; // Number of iterations when doing state propagation
        int N_; // Number of cols/rows in the workspace grid
        int sampleDim_; // Dimension of each sample
        int treeSize_; // Current size of the tree. Informs where in the d_samples_ vector to add new samples.
        float width_; // Width of the workspace
        float height_; // Height of the workspace
        float cellSize_; // Size of each cell in the workspace grid
        float costToGoal_; // Cost of the goal state
        float agentLength_; // Length of the agent. Used in state propagation.
        float connThresh_;
        thrust::device_vector<bool> d_G_; // Set of samples to be expanded in current iteration.
        thrust::device_vector<bool> d_activeU_;
        thrust::device_vector<int> d_scanIdx_; // stores scan of G. ex: G = [0, 1, 0, 1, 1, 0, 1] -> scanIdx = [0,0,1,1,2,3,3]. Used to find active samples in G.
        thrust::device_vector<int> d_activeIdx_;
        thrust::device_vector<int> d_eParentIdx_; // Stores parent sample idx. Ex: d_parentIdx[10] = 3. Parent of sample 3 is 10.
        thrust::device_vector<int> d_uParentIdx_; // stores parent indeces for current unexplored iteration.
        thrust::device_vector<float> d_uConn_;
        thrust::device_vector<float> d_samples_; // Stores all samples. Size is maxSamples_ * sampleDim_.
        thrust::device_vector<float> d_uSamples_; // all unexplored samples of current iteration.
        thrust::device_vector<float> d_xGoal_;
        bool *d_G_ptr_;
        bool *d_activeU_ptr_;
        int *d_scanIdx_ptr_;
        int *d_activeIdx_ptr_;
        int *d_eParentIdx_ptr_;
        int *d_uParentIdx_ptr_;
        float *d_uConn_ptr_;
        float *d_samples_ptr_;
        float *d_xGoal_ptr_;
        float *d_uSamples_ptr_;
        float *d_costToGoal;
        
        
        

        
};

// GPU kernels:

/**
 * @brief Populates activeGIdx with the indeces of the samples in G.
 *  EX: G = [0, 1, 0, 1, 1, 0, 1] -> activeGIdx = [1,3,4,6,0,0,0]
 */
__global__ void findInd(int numSamples, bool* G, int* scanIdx, int* activeGIdx);

/**
 * @brief Expands each sample in G. 
 * Adds new samples to eUnexplored. Calculates connectivity score for new samples. 
 * Moves G samples to eClosed.
 */
__global__ void propagateG(float* xGoal, float* uSamples, float* samples, bool* activeU, bool* G, float* uConn, int* uParentIdx, int* activeIdx_G, int activeSize_G, int treeSize, int sampleDim, float agentLength, int numDisc, curandState* states, float connThresh);
__global__ void expandEOpen(bool* eUnexplored, bool* eClosed, bool* eOpen, float* eConn, int* activeEUnexplored_Idx, int size_activeEUnexplored, float connThresh);
__global__ void expandG(float* samples, float* uSamples, int* activeIdx, int* uParentIdx, int* tParentIdx, bool* G, bool* activeU, int activeSize, int treeSize);
__global__ void initCurandStates(curandState* states, int numStates, int seed);

// GPU device functions:

/**
 * @brief Populates X1 with new sample. 
 *
 */
__device__ void propagateState(float* x0, float* x1, int numDisc, float agentLength, curandState* state);

/**
 * @brief Calculates connectivity score for a sample.
 *
 */
__device__ float calculateConnectivity(float* x, float* xGoal);

__device__ bool inCollision(float* x0, float* x1);
